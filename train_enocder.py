import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from encoder import Transformer_Encoder, Transformer_Decoder


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symlog_squared_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = symlog(pred) - symlog(target)
    return torch.mean(diff * diff)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return torch.mean(-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()))


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VideoDistanceDataset(Dataset):
    def __init__(self, video_dir: Path, distance_dir: Path, max_frames: int | None = None, image_size: int = 256):
        self.video_files = sorted([p for p in video_dir.glob("*.mp4")], key=lambda p: int(p.stem))
        self.distance_dir = distance_dir
        self.max_frames = max_frames
        self.image_size = image_size

    def __len__(self):
        return len(self.video_files)

    def _load_video(self, path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(path))
        frames = []
        count = 0
        while cap.isOpened():
            try:
                ret, frame = cap.read()
            except SystemError as exc:
                cap.release()
                raise RuntimeError(f"Failed to read frame from {path}: {exc}") from exc
            if not ret or (self.max_frames and count >= self.max_frames):
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            frame_tensor = torch.as_tensor(frame, dtype=torch.uint8).permute(2, 0, 1).contiguous()
            frames.append(frame_tensor)
            count += 1
        cap.release()
        if not frames:
            raise ValueError(f"Video {path} has no frames")
        return torch.stack(frames, dim=0)

    def _load_distances(self, idx: int) -> torch.Tensor:
        npz_path = self.distance_dir / f"{idx}.npz"
        with np.load(npz_path) as data:
            if "distances" not in data:
                raise KeyError(f"{npz_path} missing 'distances' array")
            distances = torch.from_numpy(data["distances"]).float()
        return distances

    def __getitem__(self, index):
        video_path = self.video_files[index]
        idx = int(video_path.stem)
        video = self._load_video(video_path)
        distances = self._load_distances(idx)
        if self.max_frames is not None:
            distances = distances[: video.shape[0]]
        if video.shape[0] != distances.shape[0]:
            raise ValueError(f"Frame count mismatch for {idx}: {video.shape[0]} vs {distances.shape[0]}")
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        return video, distances


def collate_full_sequence(batch):
    videos, distances = zip(*batch)
    lengths = [v.shape[1] for v in videos]
    if len(set(lengths)) != 1:
        raise ValueError("All sequences must have same length; pad first")
    video_tensor = torch.stack(videos, dim=0)
    distance_tensor = torch.stack(distances, dim=0)
    return video_tensor, distance_tensor


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    video_dir = Path(args.video_dir)
    distance_dir = Path(args.distance_dir)

    dataset = VideoDistanceDataset(video_dir, distance_dir, max_frames=args.max_frames, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_full_sequence)

    encoder = Transformer_Encoder(
        n_landmarks=args.n_landmarks,
        img_dim=128 * 4 * 4,
        n_embed=args.n_embed,
        num_heads=args.num_heads,
        n_layer=args.n_layers,
        encoder_context_window=args.context_window,
        device=device,
        dtype=dtype,
        use_kv_cache=False,
    )

    decoder = Transformer_Decoder(
        n_landmarks=args.n_landmarks,
        img_dim=128 * 4 * 4,
        n_embed=args.n_embed,
        num_heads=args.num_heads,
        n_layer=args.n_layers,
        encoder_context_window=args.context_window,
        device=device,
        dtype=dtype,
        use_kv_cache=False,
    )

    encoder.train()
    decoder.train()

    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

    scaler = torch.amp.GradScaler(enabled=device.type == "cuda") if hasattr(torch, "amp") else None

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for video, distances in dataloader:
            optimizer.zero_grad(set_to_none=True)
            video = video.to(device=device, dtype=torch.float32) / 255.0
            if dtype != torch.float32:
                video = video.to(dtype=dtype)
            distances = distances.to(device=device, dtype=dtype)

            video_for_encoder = video.permute(0, 1, 2, 4, 3)  # (B, C, T, W, H)
            video_target = video.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=dtype):
                mu, logvar, img_T, lm_T = encoder(video_for_encoder, distances)
                z = reparameterize(mu, logvar)
                recon_landmarks, recon_video = decoder(z, img_T, lm_T)

                kld = kl_divergence(mu, logvar)
                img_loss = symlog_squared_error(recon_video, video_target)
                landmark_loss = F.mse_loss(recon_landmarks, distances)

                loss = args.beta_kld * kld + args.beta_img * img_loss + args.beta_landmarks * landmark_loss

            if scaler and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f}")

    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "args": vars(args),
    }, args.output)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer encoder-decoder on video and landmark distances")
    parser.add_argument("--video_dir", type=str, default="dataset/videos")
    parser.add_argument("--distance_dir", type=str, default="dataset/distances")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_embed", type=int, default=384)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--context_window", type=int, default=4096)
    parser.add_argument("--n_landmarks", type=int, default=1106)
    parser.add_argument("--beta_kld", type=float, default=1.0)
    parser.add_argument("--beta_img", type=float, default=1.0)
    parser.add_argument("--beta_landmarks", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="encoder_model.pt")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
