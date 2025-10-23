import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from encoder import Transformer_Encoder, Transformer_Decoder
from train_enocder import VideoDistanceDataset


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
    if not sample:
        return mu
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def tensor_to_video_frames(video: torch.Tensor) -> np.ndarray:
    video = video.detach().cpu().clamp(0.0, 1.0)
    video = video.permute(0, 2, 3, 1)  # (T, H, W, C)
    return video.numpy()


def tensor_to_distances(distances: torch.Tensor) -> np.ndarray:
    return distances.detach().cpu().numpy()


def visualize_sequences(original_video: np.ndarray, reconstructed_video: np.ndarray, original_distances: np.ndarray, reconstructed_distances: np.ndarray, num_frames: int) -> None:
    frames_to_show = min(num_frames, original_video.shape[0], reconstructed_video.shape[0])

    fig, axes = plt.subplots(2, frames_to_show, figsize=(3 * frames_to_show, 6))
    for i in range(frames_to_show):
        axes[0, i].imshow(original_video[i])
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Original #{i}")

        axes[1, i].imshow(reconstructed_video[i])
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Reconstructed #{i}")

    fig.suptitle("Video Reconstruction")
    plt.tight_layout()

    plt.figure(figsize=(12, 6))
    time = np.arange(original_distances.shape[0])
    landmark_idx = 0
    plt.plot(time, original_distances[:, landmark_idx], label="Original", linewidth=2)
    plt.plot(time, reconstructed_distances[:, landmark_idx], label="Reconstructed", linestyle="--", linewidth=2)
    plt.xlabel("Frame")
    plt.ylabel(f"Landmark {landmark_idx}")
    plt.title("Landmark Distance Reconstruction")
    plt.legend()
    plt.tight_layout()
    plt.show()


def load_models(checkpoint_path: Path, device: torch.device, dtype: torch.dtype) -> tuple[Transformer_Encoder, Transformer_Decoder, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args_dict = checkpoint.get("args", {})

    n_landmarks = args_dict.get("n_landmarks", 1106)
    n_embed = args_dict.get("n_embed", 256)
    num_heads = args_dict.get("num_heads", 2)
    n_layers = args_dict.get("n_layers", 2)
    context_window = args_dict.get("context_window", 8192)

    encoder = Transformer_Encoder(
        n_landmarks=n_landmarks,
        img_dim=128 * 4 * 4,
        n_embed=n_embed,
        num_heads=num_heads,
        n_layer=n_layers,
        encoder_context_window=context_window,
        device=device,
        dtype=dtype,
        use_kv_cache=False,
    )

    decoder = Transformer_Decoder(
        n_landmarks=n_landmarks,
        img_dim=128 * 4 * 4,
        n_embed=n_embed,
        num_heads=num_heads,
        n_layer=n_layers,
        encoder_context_window=context_window,
        device=device,
        dtype=dtype,
        use_kv_cache=False,
    )

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()

    return encoder, decoder, args_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run encoder-decoder reconstruction on a sample sequence")
    parser.add_argument("--video_dir", type=Path, default=Path("dataset/videos"))
    parser.add_argument("--distance_dir", type=Path, default=Path("dataset/distances"))
    parser.add_argument("--checkpoint", type=Path, default=Path("encoder_model.pt"))
    parser.add_argument("--sample_index", type=int, default=0, help="Index of the sample sequence to visualize")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu or cuda)")
    parser.add_argument("--num_frames", type=int, default=6, help="Number of frames to visualize")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of frames per video")
    parser.add_argument("--image_size", type=int, default=256, help="Resize frames before feeding to the encoder")
    parser.add_argument("--sample_latent", action="store_true", help="Sample latent from posterior instead of using mean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16 if torch.cuda.is_available() else torch.float32

    encoder, decoder, _ = load_models(args.checkpoint, device, dtype)

    dataset = VideoDistanceDataset(
        args.video_dir,
        args.distance_dir,
        max_frames=args.max_frames,
        image_size=args.image_size,
        mode="videos",
    )

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Ensure video and distance files are available.")

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"Sample index {args.sample_index} out of range (dataset size: {len(dataset)})")

    video, distances = dataset[args.sample_index]
    video = video.unsqueeze(0).float() / 255.0  # (1, C, T, H, W)
    distances = distances.unsqueeze(0).to(dtype=dtype)

    video_for_encoder = video.permute(0, 1, 2, 4, 3).to(device=device, dtype=dtype)
    distances = distances.to(device=device)

    with torch.no_grad():
        mu, logvar, img_T, lm_T = encoder(video_for_encoder, distances)
        z = reparameterize(mu, logvar, sample=args.sample_latent)
        recon_distances, recon_video = decoder(z, img_T, lm_T)

    recon_video = recon_video.to(torch.float32)
    recon_video = symexp(recon_video)

    print(recon_video[0])

    video_target = video.permute(0, 2, 1, 3, 4).to(torch.float32)

    original_frames = tensor_to_video_frames(video_target[0])
    reconstructed_frames = tensor_to_video_frames(recon_video[0])

    original_distances = tensor_to_distances(distances[0])
    reconstructed_distances = tensor_to_distances(recon_distances[0])

    visualize_sequences(
        original_frames,
        reconstructed_frames,
        original_distances,
        reconstructed_distances,
        num_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()
