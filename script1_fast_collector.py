"""
script1_fast_collector.py
-------------------------
High-Speed Automated Sign Language Dataset Collector with 3D Data Augmentation.

Why this is 10x faster:
1. Automated Queue: You define your words upfront (e.g., HELLO, YES, NO, THANKS, HELP).
2. Hands-Free Workflow: Automatically guides you through each sign with a 2-second transition.
3. Rapid 1-Second Capture: You only need to hold each gesture for 30 frames (~1 second).
4. Geometric Data Augmentation: Automatically applies 3D micro-rotations and scaling to expand
   30 frames into 150 diverse training samples per gesture.
   -> Record 6 full signs in under 30 seconds total!
"""

import os
# Suppress Google/MediaPipe C++ telemetry and logging
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import csv
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

DATASET_FILE = "dataset.csv"
RAW_FRAMES_PER_SIGN = 30       # Only 30 frames (~1 sec) needed thanks to augmentation!
TRANSITION_SECONDS = 2.5       # Seconds between signs to adjust hand
WEBCAM_INDEX = 0
TARGET_FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_task_model():
    if not os.path.exists(TASK_MODEL_PATH):
        print(f"[INFO] Downloading MediaPipe hand landmarker model...")
        urllib.request.urlretrieve(TASK_MODEL_URL, TASK_MODEL_PATH)


class HandDetector:
    def __init__(self, max_hands=2):
        self.max_hands = max_hands
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.mode = "solutions"
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.detector = self.mp_hands.Hands(
                static_image_mode=False, max_num_hands=self.max_hands, min_detection_confidence=0.7
            )
        else:
            self.mode = "tasks"
            ensure_task_model()
            from mediapipe.tasks.python import vision
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python.vision import drawing_utils, HandLandmarksConnections

            self.drawing_utils = drawing_utils
            self.connections = HandLandmarksConnections.HAND_CONNECTIONS
            base_options = mp_python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_hands=self.max_hands
            )
            self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr):
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.mode == "solutions":
            results = self.detector.process(rgb_frame)
            return results.multi_hand_landmarks if results.multi_hand_landmarks else None
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)
            return results.hand_landmarks if results.hand_landmarks else None

    def draw_landmarks(self, frame_bgr, hand_landmarks):
        if self.mode == "solutions":
            self.mp_drawing.draw_landmarks(
                frame_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        else:
            self.drawing_utils.draw_landmarks(frame_bgr, hand_landmarks, self.connections)

    def close(self):
        if self.mode == "solutions":
            self.detector.close()


def normalize_landmarks(hand_landmarks) -> np.ndarray:
    raw_coords = []
    landmark_list = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
    for lm in landmark_list:
        raw_coords.extend([lm.x, lm.y, lm.z])
    coords = np.array(raw_coords, dtype=np.float32).reshape(21, 3)
    wrist = coords[0, :]
    relative_coords = coords - wrist
    max_val = np.max(np.abs(relative_coords))
    return (relative_coords / max_val if max_val > 1e-6 else relative_coords).flatten()


def augment_landmark_vector(vector_63: np.ndarray) -> list:
    """
    Synthesizes 5 distinct variations from a single landmark vector:
    1. Original normalized pose
    2. Slight clockwise rotation (~4.5 degrees)
    3. Slight counter-clockwise rotation (~4.5 degrees)
    4. Small scale expansion (+5%)
    5. Small scale contraction (-5%)
    """
    augmented = [vector_63]
    coords = vector_63.reshape(21, 3)

    # 3D In-plane Micro-rotations
    for angle in [-0.08, 0.08]:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
        rotated = np.dot(coords, rot_matrix)
        augmented.append(rotated.flatten())

    # Distance/Scale jitter
    augmented.append((coords * 1.05).flatten())
    augmented.append((coords * 0.95).flatten())

    return augmented


def initialize_csv(filepath: str):
    if not os.path.exists(filepath):
        header = ["label"] + [f"{axis}_{i}" for i in range(21) for axis in ("x", "y", "z")]
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def append_data_to_csv(filepath: str, label: str, samples: list):
    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for s in samples:
            writer.writerow([label] + s.tolist())


def main():
    initialize_csv(DATASET_FILE)

    print("\n" + "=" * 65)
    print(" HIGH-SPEED AUTOMATED SIGN DATASET COLLECTOR")
    print("=" * 65)
    print("This tool lets you record multiple signs hands-free in under 1 minute.")
    print("Enter the signs you want to record, separated by commas.")
    print("Example: HELLO, YES, NO, THANKS, HELP, PLEASE")
    print("=" * 65 + "\n")

    user_input = input("Enter signs to record (or press ENTER for default [HELLO, YES, NO, THANKS, HELP]): ").strip()
    if not user_input:
        signs_to_record = ["HELLO", "YES", "NO", "THANKS", "HELP"]
    else:
        signs_to_record = [s.strip().upper() for s in user_input.split(",") if s.strip()]

    print(f"\n[QUEUE] Will record {len(signs_to_record)} signs: {', '.join(signs_to_record)}")
    print("Opening webcam. Look at the camera window to follow the countdown prompts!\n")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {WEBCAM_INDEX}.")
        return

    detector = HandDetector()

    current_idx = 0
    phase = "TRANSITION"  # "TRANSITION" or "RECORDING" or "FINISHED"
    phase_start_time = time.time()
    raw_samples = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            now = time.time()

            hands_detected = detector.process(frame)
            hand_in_view = bool(hands_detected)

            if hand_in_view:
                for hand in hands_detected:
                    detector.draw_landmarks(frame, hand)

            if phase == "TRANSITION":
                target_sign = signs_to_record[current_idx]
                elapsed = now - phase_start_time
                remaining = max(0.0, TRANSITION_SECONDS - elapsed)

                # Overlay banner
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h // 2 - 90), (w, h // 2 + 80), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

                cv2.putText(frame, f"GET READY FOR SIGN {current_idx + 1}/{len(signs_to_record)}:",
                            (w // 2 - 200, h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(frame, f"'{target_sign}'",
                            (w // 2 - 120, h // 2 + 10), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 255), 3)
                cv2.putText(frame, f"Starting in: {remaining:.1f}s",
                            (w // 2 - 90, h // 2 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

                if remaining <= 0:
                    phase = "RECORDING"
                    raw_samples = []
                    phase_start_time = now

            elif phase == "RECORDING":
                target_sign = signs_to_record[current_idx]

                if hand_in_view:
                    primary_hand = hands_detected[0]
                    norm_vec = normalize_landmarks(primary_hand)
                    raw_samples.append(norm_vec)

                    count = len(raw_samples)
                    cv2.putText(frame, f"HOLD GESTURE '{target_sign}': {count}/{RAW_FRAMES_PER_SIGN}",
                                (20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 0), 2)

                    bar_w = int((count / RAW_FRAMES_PER_SIGN) * (w - 40))
                    cv2.rectangle(frame, (20, 60), (20 + bar_w, 75), (0, 255, 0), -1)
                    cv2.rectangle(frame, (20, 60), (w - 20, 75), (200, 200, 200), 2)

                    if count >= RAW_FRAMES_PER_SIGN:
                        # Augment and save!
                        all_augmented = []
                        for vec in raw_samples:
                            all_augmented.extend(augment_landmark_vector(vec))

                        append_data_to_csv(DATASET_FILE, target_sign, all_augmented)
                        print(f"[SUCCESS] Saved {len(all_augmented)} samples for '{target_sign}' (including 3D augmentations)!")

                        current_idx += 1
                        if current_idx >= len(signs_to_record):
                            phase = "FINISHED"
                            phase_start_time = now
                        else:
                            phase = "TRANSITION"
                            phase_start_time = now
                else:
                    cv2.putText(frame, f"PLACE HAND IN VIEW FOR '{target_sign}'!",
                                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            elif phase == "FINISHED":
                cv2.rectangle(frame, (0, h // 2 - 80), (w, h // 2 + 60), (0, 100, 0), -1)
                cv2.putText(frame, "ALL SIGNS RECORDED SUCCESSFULLY!",
                            (w // 2 - 250, h // 2 - 20), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"Added {len(signs_to_record)} signs to dataset.csv. Press [Q] to Exit",
                            (w // 2 - 230, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 255, 220), 1)

                if now - phase_start_time > 4.0:
                    break

            cv2.imshow("Fast Sign Data Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:
                break

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Fast collection session completed cleanly.")


if __name__ == "__main__":
    main()
