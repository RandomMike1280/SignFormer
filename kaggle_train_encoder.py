from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import shutil
import cv2
import numpy as np
from datasets import Video, load_dataset
import mediapipe as mp
from tqdm.auto import tqdm
from train_enocder import train

NUM_FACE_LANDMARKS = 478
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
TOTAL_LANDMARKS = NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS + 2 * NUM_HAND_LANDMARKS


def extract_landmarks(results):
    arr = np.zeros((TOTAL_LANDMARKS, 2), dtype=np.float32)
    segments = (
        (results.pose_landmarks, 0, NUM_POSE_LANDMARKS),
        (results.face_landmarks, NUM_POSE_LANDMARKS, NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS),
        (
            results.left_hand_landmarks,
            NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS,
            NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS + NUM_HAND_LANDMARKS,
        ),
        (
            results.right_hand_landmarks,
            NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS + NUM_HAND_LANDMARKS,
            TOTAL_LANDMARKS,
        ),
    )
    for data, start, end in segments:
        if data:
            arr[start:end] = np.array([(lm.x, lm.y) for lm in data.landmark], dtype=np.float32)
    return arr


def create_distance_indices():
    idx1 = [0, 1, 2]
    idx2 = [1, 2, 0]
    for i in range(2, TOTAL_LANDMARKS):
        idx1.extend([i, i])
        idx2.extend([i - 2, i - 1])
    idx1.append(0)
    idx2.append(TOTAL_LANDMARKS - 1)
    return np.array(idx1, dtype=np.int64), np.array(idx2, dtype=np.int64)


def calculate_distances(landmarks_xy, indices1, indices2):
    points1 = landmarks_xy[indices1]
    points2 = landmarks_xy[indices2]
    distances = np.linalg.norm(points1 - points2, axis=1)
    mask = np.all(points1 == 0, axis=1) | np.all(points2 == 0, axis=1)
    distances[mask] = 0.0
    return distances.astype(np.float32)


def compute_distances_for_video(video_path, holistic, indices1, indices2, max_frames=None):
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    count = 0
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        landmarks = extract_landmarks(results)
        frames.append(calculate_distances(landmarks, indices1, indices2))
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    capture.release()
    if not frames:
        return np.zeros((0, indices1.shape[0]), dtype=np.float32)
    return np.stack(frames).astype(np.float32)


@dataclass
class EncoderTrainConfig:
    dataset_root: Path
    split: str = "train"
    overwrite: bool = False
    max_frames: int | None = None
    image_size: int = 256
    batch_size: int = 1
    epochs: int = 10
    lr: float = 3e-4
    n_embed: int = 256
    num_heads: int = 2
    n_layers: int = 2
    context_window: int = 8192
    n_landmarks: int = 1106
    beta_kld: float = 1.0
    beta_img: float = 1.0
    beta_landmarks: float = 1.0
    frame_epochs: int = 0
    frame_batch_size: int | None = None
    output: str = "encoder_model.pt"
    test: bool = False


def prepare_vsl_signs_dataset(config: EncoderTrainConfig):
    dataset = load_dataset("NoOne1280/VSL-Signs", split=config.split)
    dataset = dataset.cast_column("video", Video(decode=False))
    video_dir = config.dataset_root / "videos"
    distance_dir = config.dataset_root / "distances"
    video_dir.mkdir(parents=True, exist_ok=True)
    distance_dir.mkdir(parents=True, exist_ok=True)
    indices1, indices2 = create_distance_indices()
    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
    ) as holistic:
        for idx, sample in enumerate(tqdm(dataset, total=len(dataset))):
            src_path = Path(sample["video"]["path"])
            video_path = video_dir / f"{idx}.mp4"
            distance_path = distance_dir / f"{idx}.npz"
            if config.overwrite or not video_path.exists():
                shutil.copy2(src_path, video_path)
            if config.overwrite or not distance_path.exists():
                distances = compute_distances_for_video(
                    video_path,
                    holistic,
                    indices1,
                    indices2,
                    config.max_frames,
                )
                np.savez_compressed(distance_path, distances=distances)
    return video_dir, distance_dir


def train_encoder_from_hf(config: EncoderTrainConfig):
    video_dir, distance_dir = prepare_vsl_signs_dataset(config)
    args = SimpleNamespace(
        video_dir=str(video_dir),
        distance_dir=str(distance_dir),
        batch_size=config.batch_size,
        epochs=config.epochs,
        lr=config.lr,
        n_embed=config.n_embed,
        num_heads=config.num_heads,
        n_layers=config.n_layers,
        context_window=config.context_window,
        n_landmarks=config.n_landmarks,
        beta_kld=config.beta_kld,
        beta_img=config.beta_img,
        beta_landmarks=config.beta_landmarks,
        output=config.output,
        max_frames=config.max_frames,
        image_size=config.image_size,
        frame_epochs=config.frame_epochs,
        frame_batch_size=config.frame_batch_size,
        test=config.test,
    )
    train(args)
