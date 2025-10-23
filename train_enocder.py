import argparse
import math
import os
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from encoder import Transformer_Encoder, Transformer_Decoder


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def symlog_squared_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # The network would learn the transformed version of its target y,\
    # and to read out the predictions of the network we apply the inverse transformation with symexp
    diff = pred - symlog(target)
    return torch.mean(diff * diff)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return torch.mean(-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()))


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VideoDistanceDataset(Dataset):
    def __init__(
        self,
        video_dir: Path,
        distance_dir: Path,
        max_frames: int | None = None,
        image_size: int = 256,
        mode: str = "videos",
        subset_fraction: float | None = None,
    ):
        self.max_frames = max_frames
        self.image_size = image_size
        self.mode = mode
        raw_metadata = []
        frame_counts = []
        for video_path in sorted([p for p in video_dir.glob("*.mp4")], key=lambda p: int(p.stem)):
            idx = int(video_path.stem)
            distance_path = distance_dir / f"{idx}.npz"
            with np.load(distance_path, mmap_mode="r") as data:
                if "distances" not in data:
                    raise KeyError(f"{distance_path} missing 'distances' array")
                num_frames = data["distances"].shape[0]
            if self.max_frames is not None:
                num_frames = min(num_frames, self.max_frames)
            if num_frames == 0:
                continue
            raw_metadata.append({
                "idx": idx,
                "video_path": video_path,
                "distance_path": distance_path,
                "num_frames": num_frames,
            })
            frame_counts.append(num_frames)

        if subset_fraction is not None and raw_metadata:
            limit = max(1, math.ceil(len(raw_metadata) * subset_fraction))
            raw_metadata = raw_metadata[:limit]
            frame_counts = frame_counts[:limit]

        self.video_metadata = raw_metadata

        if self.mode == "frames":
            frame_offsets = []
            total_frames = 0
            for count in frame_counts:
                total_frames += count
                frame_offsets.append(total_frames)
            self.frame_offsets = frame_offsets
        else:
            self.frame_offsets = None

    def __len__(self):
        if self.mode == "frames":
            return self.frame_offsets[-1] if self.frame_offsets else 0
        return len(self.video_metadata)

    def _prepare_frame(self, frame: np.ndarray) -> torch.Tensor:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return torch.as_tensor(frame, dtype=torch.uint8).permute(2, 0, 1).contiguous()

    def _load_video(self, meta: dict) -> torch.Tensor:
        cap = cv2.VideoCapture(str(meta["video_path"]))
        frames = []
        count = 0
        while cap.isOpened():
            try:
                ret, frame = cap.read()
            except SystemError as exc:
                cap.release()
                raise RuntimeError(f"Failed to read frame from {meta['video_path']}: {exc}") from exc
            if not ret or count >= meta["num_frames"]:
                break
            frames.append(self._prepare_frame(frame))
            count += 1
        cap.release()
        if not frames:
            raise ValueError(f"Video {meta['video_path']} has no frames")
        return torch.stack(frames, dim=0)

    def _load_frame(self, meta: dict, frame_index: int) -> torch.Tensor:
        cap = cv2.VideoCapture(str(meta["video_path"]))
        if frame_index >= meta["num_frames"]:
            cap.release()
            raise IndexError(f"Frame index {frame_index} out of range for video {meta['idx']}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_index} from {meta['video_path']}")
        return self._prepare_frame(frame)

    def _load_distances(self, meta: dict, frame_slice=None) -> torch.Tensor:
        with np.load(meta["distance_path"], mmap_mode="r") as data:
            if "distances" not in data:
                raise KeyError(f"{meta['distance_path']} missing 'distances' array")
            distances = data["distances"]
            if frame_slice is None:
                distances_slice = distances[: meta["num_frames"]]
            else:
                distances_slice = distances[frame_slice]
            distances_array = np.array(distances_slice, dtype=np.float32)
        return torch.from_numpy(distances_array).float()

    def __getitem__(self, index):
        if self.mode == "frames":
            if not self.frame_offsets:
                raise IndexError("Frame dataset is empty")
            frame_pos = index
            video_idx = bisect_left(self.frame_offsets, frame_pos + 1)
            previous = 0 if video_idx == 0 else self.frame_offsets[video_idx - 1]
            frame_in_video = frame_pos - previous
            meta = self.video_metadata[video_idx]
            frame = self._load_frame(meta, frame_in_video)
            distance = self._load_distances(meta, frame_in_video)
            return frame, distance
        meta = self.video_metadata[index]
        video = self._load_video(meta)
        distances = self._load_distances(meta)
        if video.shape[0] != distances.shape[0]:
            raise ValueError(f"Frame count mismatch for {meta['idx']}: {video.shape[0]} vs {distances.shape[0]}")
        video = video.permute(1, 0, 2, 3)
        return video, distances


def collate_full_sequence(batch):
    videos, distances = zip(*batch)
    lengths = [v.shape[1] for v in videos]
    if len(set(lengths)) != 1:
        raise ValueError("All sequences must have same length; pad first")
    video_tensor = torch.stack(videos, dim=0)
    distance_tensor = torch.stack(distances, dim=0)
    return video_tensor, distance_tensor


def collate_single_frames(batch):
    frames, distances = zip(*batch)
    frame_tensor = torch.stack(frames, dim=0).unsqueeze(2)
    distance_tensor = torch.stack(distances, dim=0)
    if distance_tensor.ndim == 2:
        distance_tensor = distance_tensor.unsqueeze(1)
    return frame_tensor, distance_tensor


def train(args, rank: int = 0, world_size: int = 1, distributed: bool = False):
    if distributed:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    video_dir = Path(args.video_dir)
    distance_dir = Path(args.distance_dir)

    if args.test:
        args.batch_size = 1
        args.epochs = 1 if args.epochs == 0 else args.epochs

    subset_fraction = 0.1 if args.test else None

    video_dataset = VideoDistanceDataset(
        video_dir,
        distance_dir,
        max_frames=args.max_frames,
        image_size=args.image_size,
        mode="videos",
        subset_fraction=subset_fraction,
    )
    video_sampler = DistributedSampler(video_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    video_loader = DataLoader(
        video_dataset,
        batch_size=args.batch_size,
        shuffle=not distributed,
        sampler=video_sampler,
        collate_fn=collate_full_sequence,
    )

    frame_loader = None
    frame_epochs = min(args.frame_epochs, args.epochs)
    if frame_epochs > 0:
        frame_dataset = VideoDistanceDataset(
            video_dir,
            distance_dir,
            max_frames=args.max_frames,
            image_size=args.image_size,
            mode="frames",
            subset_fraction=subset_fraction,
        )
        if len(frame_dataset) > 0:
            frame_sampler = DistributedSampler(frame_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
            frame_loader = DataLoader(
                frame_dataset,
                batch_size=args.frame_batch_size or args.batch_size,
                shuffle=not distributed,
                sampler=frame_sampler,
                collate_fn=collate_single_frames,
            )
        else:
            frame_sampler = None
    else:
        frame_sampler = None
        frame_epochs = 0

    if len(video_dataset) == 0:
        raise ValueError("No videos found for training")

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

    if distributed:
        encoder = DDP(encoder, device_ids=[rank], output_device=rank)
        decoder = DDP(decoder, device_ids=[rank], output_device=rank)
        encoder.train()
        decoder.train()

    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

    scaler = torch.amp.GradScaler(enabled=device.type == "cuda") if hasattr(torch, "amp") else None

    printed_shapes = False

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        if epoch < frame_epochs and frame_loader is not None:
            current_loader = frame_loader
            stage = "frames"
            current_sampler = frame_sampler
        else:
            current_loader = video_loader
            stage = "videos"
            current_sampler = video_sampler
        if distributed and current_sampler is not None:
            current_sampler.set_epoch(epoch)
        if args.test:
            if rank == 0:
                print(f"Num Epochs: {args.epochs}")
                print(f"Num Frame Epochs: {frame_epochs}")
                print(f"Length loader: {len(current_loader)}")
        for video, distances in current_loader:
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

                if args.test and not printed_shapes and rank == 0:
                    print("Input video_for_encoder shape:", tuple(video_for_encoder.shape))
                    print("Input distances shape:", tuple(distances.shape))
                    print("Encoder outputs - mu shape:", tuple(mu.shape), "logvar shape:", tuple(logvar.shape))
                    print("Decoder outputs - recon_landmarks shape:", tuple(recon_landmarks.shape), "recon_video shape:", tuple(recon_video.shape))
                    printed_shapes = True

                kld = kl_divergence(mu, logvar)
                img_loss = symlog_squared_error(recon_video, video_target)
                landmark_loss = F.mse_loss(recon_landmarks, distances)

                loss = args.beta_kld * kld + args.beta_img * img_loss + args.beta_landmarks * landmark_loss

            if scaler and scaler.is_enabled():
                if args.test:
                    if rank == 0:
                        print("Performing backward pass with gradient scaling.")
                    pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.test:
                    # print("Performing backward pass without gradient scaling.")
                    pass
                loss.backward()
                optimizer.step()
            if args.test:
                if rank == 0:
                    print(f"Loss: {kld.item():.4f} | {img_loss.item():.4f} | {landmark_loss.item():.4f}")
            epoch_loss += loss.item()

        if distributed:
            loss_tensor = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = loss_tensor.item()
            denom = len(current_loader) * world_size
        else:
            denom = len(current_loader)
        avg_loss = epoch_loss / denom
        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs} [{stage}] - Loss: {avg_loss:.4f}")

    if not distributed or rank == 0:
        torch.save({
            "encoder": encoder.module.state_dict() if distributed else encoder.state_dict(),
            "decoder": decoder.module.state_dict() if distributed else decoder.state_dict(),
            "args": vars(args),
        }, args.output)


def train_worker(rank: int, world_size: int, args):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        train(args, rank=rank, world_size=world_size, distributed=True)
    finally:
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer encoder-decoder on video and landmark distances")
    parser.add_argument("--video_dir", type=str, default="dataset/videos")
    parser.add_argument("--distance_dir", type=str, default="dataset/distances")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_embed", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--context_window", type=int, default=8192)
    parser.add_argument("--n_landmarks", type=int, default=1106)
    parser.add_argument("--beta_kld", type=float, default=1.0)
    parser.add_argument("--beta_img", type=float, default=1.0)
    parser.add_argument("--beta_landmarks", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="encoder_model.pt")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--frame_epochs", type=int, default=0)
    parser.add_argument("--frame_batch_size", type=int, default=None)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        if world_size > 1:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
            mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
        else:
            train(args)
    else:
        train(args)
