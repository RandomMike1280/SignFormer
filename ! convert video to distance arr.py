import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Constants and Initialization ---
NUM_FACE_LANDMARKS = 478
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
TOTAL_LANDMARKS = NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS + 2 * NUM_HAND_LANDMARKS

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

small_dots = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
small_lines = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)

# --- Core Helper Functions (Unchanged) ---
def extract_landmarks(results) -> np.ndarray:
    """Extracts landmark coordinates (x, y) into a single NumPy array."""
    landmarks_array = np.zeros((TOTAL_LANDMARKS, 2), dtype=np.float32)
    sources = [
        (results.pose_landmarks, 0, NUM_POSE_LANDMARKS),
        (results.face_landmarks, NUM_POSE_LANDMARKS, NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS),
        (results.left_hand_landmarks, NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS, NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS + NUM_HAND_LANDMARKS),
        (results.right_hand_landmarks, NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS + NUM_HAND_LANDMARKS, TOTAL_LANDMARKS)
    ]
    for landmarks, start, end in sources:
        if landmarks:
            coords = [(lm.x, lm.y) for lm in landmarks.landmark]
            landmarks_array[start:end] = np.array(coords, dtype=np.float32)
    return landmarks_array

def create_distance_indices():
    """Pre-computes the pairs of indices needed for the distance calculation."""
    indices1, indices2 = [], []
    indices1.extend([0, 1, 2])
    indices2.extend([1, 2, 0])
    for i in range(2, TOTAL_LANDMARKS):
        indices1.extend([i, i])
        indices2.extend([i - 2, i - 1])
    indices1.append(0)
    indices2.append(TOTAL_LANDMARKS - 1)
    return np.array(indices1), np.array(indices2)

def calculate_distances_vectorized(landmarks_xy: np.ndarray, indices1: np.ndarray, indices2: np.ndarray) -> np.ndarray:
    """Calculates distances between specified landmark pairs in a vectorized manner."""
    points1 = landmarks_xy[indices1]
    points2 = landmarks_xy[indices2]
    distances = np.linalg.norm(points1 - points2, axis=1)
    zero_mask = np.all(points1 == 0, axis=1) | np.all(points2 == 0, axis=1)
    distances[zero_mask] = 0
    return distances

# --- Main Processing Function ---
def process_video(input_path: str, output_video_path: str, output_data_path: str):
    """
    Processes a video file to extract landmark distances and render landmarks.

    Args:
        input_path (str): Path to the source .mp4 video.
        output_video_path (str): Path to save the output .mp4 video with rendered landmarks.
        output_data_path (str): Path to save the calculated distances as a .npz file.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        print(f"Error: Input video not found at {input_path}")
        return

    # Pre-calculate indices once
    distance_indices1, distance_indices2 = create_distance_indices()
    
    # Setup video capture
    cap = cv2.VideoCapture(str(input_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    all_frame_distances = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
    ) as holistic:
        
        # Process video frame by frame
        for _ in tqdm(range(total_frames), desc=f"Processing {input_path.name}"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- Core Logic ---
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            landmarks_xy = extract_landmarks(results)
            distances_np = calculate_distances_vectorized(
                landmarks_xy, distance_indices1, distance_indices2
            )
            all_frame_distances.append(distances_np)

            # --- Drawing Logic ---
            drawing_specs = [
                (results.pose_landmarks, mp_holistic.POSE_CONNECTIONS),
                (results.face_landmarks, mp_holistic.FACEMESH_TESSELATION),
                (results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS),
                (results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS),
            ]
            for landmarks, connections in drawing_specs:
                if landmarks:
                    mp_drawing.draw_landmarks(
                        frame, landmarks, connections, small_dots, small_lines
                    )
            
            writer.write(frame)

    # --- Cleanup and Saving ---
    cap.release()
    writer.release()
    print(f"\n✅ Video processing complete. Rendered video saved to: {output_video_path}")

    # Convert list of arrays to a single 2D array and save
    final_distances_array = np.array(all_frame_distances)
    np.savez_compressed(output_data_path, distances=final_distances_array)
    print(f"✅ Landmark distances saved to: {output_data_path}")
    print(f"Data shape: {final_distances_array.shape} (frames, distances_per_frame)")

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Define file paths
    for name in [
        r'2', r'3', r'4',r'5', r'6', r'7', r'8', r'9', r'10'
    ]:
        INPUT_VIDEO = fr"dataset\videos\{name}.mp4"
        OUTPUT_VIDEO = fr"dataset\visualizations\{name}.mp4"
        OUTPUT_DATA = fr"dataset\distances\{name}.npz"

        if not Path(INPUT_VIDEO).exists():
            raise Exception('Path does not exist')

        # 3. Run the processing function
        process_video(
            input_path=INPUT_VIDEO,
            output_video_path=OUTPUT_VIDEO,
            output_data_path=OUTPUT_DATA
        )

        # 4. (Optional) Verify the output data
        print("\n--- Verifying Output ---")
        try:
            data = np.load(OUTPUT_DATA)
            distances = data['distances']
            print(f"Successfully loaded data from '{OUTPUT_DATA}'")
            print(f"Array shape: {distances.shape}")
            print("Verification complete.")
        except Exception as e:
            print(f"Could not verify output file: {e}")