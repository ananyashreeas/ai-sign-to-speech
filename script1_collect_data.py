"""
script1_collect_data.py
-----------------------
Collects normalized hand landmark data using MediaPipe and OpenCV
to build a structured training dataset for sign language classification.

Features:
- Completely GUI-driven: type sign names directly ON SCREEN inside the OpenCV window.
  (No need to switch to terminal or deal with console freezes/telemetry logs).
- Automatically suppresses background MediaPipe telemetry / clearcut logs.
- Detects 21 3D hand landmarks and normalizes relative to wrist (landmark 0).
- Visual countdown (3... 2... 1...) and live progress bar for 100 frames.
- Automatically saves data directly to "dataset.csv".
"""

import os
# Suppress Google/MediaPipe C++ telemetry and logging BEFORE importing mediapipe
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import csv
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
DATASET_FILE = "dataset.csv"
FRAMES_PER_SIGN = 100
WEBCAM_INDEX = 0
TARGET_FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_task_model():
    """Downloads the official MediaPipe Hand Landmarker model file if not present."""
    if not os.path.exists(TASK_MODEL_PATH):
        print(f"[INFO] Downloading MediaPipe hand landmarker model to '{TASK_MODEL_PATH}'...")
        urllib.request.urlretrieve(TASK_MODEL_URL, TASK_MODEL_PATH)
        print("[INFO] Download completed.")


class HandDetector:
    """
    Unified Hand Detector compatible with modern MediaPipe Tasks and legacy Solutions.
    """

    def __init__(self, max_hands=2):
        self.max_hands = max_hands
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.mode = "solutions"
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_hands,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
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
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.max_hands,
                min_hand_detection_confidence=0.7
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
                frame_bgr,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
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
    if max_val > 1e-6:
        normalized_coords = relative_coords / max_val
    else:
        normalized_coords = relative_coords

    return normalized_coords.flatten()


def initialize_csv(filepath: str):
    if not os.path.exists(filepath):
        header = ["label"]
        for i in range(21):
            header.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_data_to_csv(filepath: str, label: str, samples: list):
    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for sample in samples:
            writer.writerow([label] + sample.tolist())


def get_existing_classes(filepath: str) -> list:
    if not os.path.exists(filepath):
        return []
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        if not df.empty and "label" in df.columns:
            return list(df["label"].unique())
    except Exception:
        pass
    return []


def main():
    initialize_csv(DATASET_FILE)
    existing_signs = get_existing_classes(DATASET_FILE)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {WEBCAM_INDEX}.")
        return

    detector = HandDetector()

    # Pre-select initial label or default
    current_label = "YES" if "HELLO" in existing_signs else "HELLO"
    
    # State flags
    is_recording = False
    countdown_active = False
    countdown_start_time = 0.0
    collected_samples = []
    
    # On-screen text input state
    is_typing_label = False
    typed_buffer = ""
    status_message = f"Ready. Current sign: {current_label}"

    print("\n" + "=" * 60)
    print(" SIGN LANGUAGE DATA COLLECTOR (ON-SCREEN GUI)")
    print("=" * 60)
    print("Existing signs in dataset:", existing_signs if existing_signs else "None yet")
    print("Controls:")
    print("  [SPACE] / [S] : Start recording 100 frames")
    print("  [N]           : Type new sign name directly on screen")
    print("  [Q] / [ESC]   : Quit program")
    print("=" * 60 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Detect landmarks
            hands_detected = detector.process(frame)
            hand_in_view = bool(hands_detected)

            if hand_in_view:
                for hand in hands_detected:
                    detector.draw_landmarks(frame, hand)

            # --- State A: On-Screen Text Input Modal ---
            if is_typing_label:
                overlay = frame.copy()
                cv2.rectangle(overlay, (40, h // 2 - 80), (w - 40, h // 2 + 70), (30, 30, 30), -1)
                cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                cv2.rectangle(frame, (40, h // 2 - 80), (w - 40, h // 2 + 70), (0, 200, 255), 2)

                cv2.putText(
                    frame, "TYPE NEW SIGN NAME & PRESS ENTER:",
                    (60, h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA
                )
                # Show typed buffer with blinking cursor
                cursor = "_" if int(time.time() * 2) % 2 == 0 else ""
                cv2.putText(
                    frame, f"> {typed_buffer}{cursor}",
                    (60, h // 2 + 5), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA
                )
                cv2.putText(
                    frame, "[ENTER] Save  |  [ESC] Cancel",
                    (60, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA
                )

            # --- State B: 3-Second Pre-record Countdown ---
            elif countdown_active:
                elapsed = time.time() - countdown_start_time
                remaining = int(3.5 - elapsed)
                if remaining > 0:
                    cv2.putText(
                        frame, f"GET READY: {remaining}",
                        (w // 2 - 160, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 140, 255), 3, cv2.LINE_AA
                    )
                else:
                    countdown_active = False
                    is_recording = True
                    collected_samples = []

            # --- State C: Active Recording (100 frames) ---
            elif is_recording:
                if hand_in_view:
                    primary_hand = hands_detected[0]
                    normalized_vector = normalize_landmarks(primary_hand)
                    collected_samples.append(normalized_vector)

                    count = len(collected_samples)
                    progress_text = f"Recording '{current_label}': {count}/{FRAMES_PER_SIGN}"
                    cv2.putText(
                        frame, progress_text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
                    )

                    bar_w = int((count / FRAMES_PER_SIGN) * (w - 40))
                    cv2.rectangle(frame, (20, 65), (20 + bar_w, 80), (0, 255, 0), -1)
                    cv2.rectangle(frame, (20, 65), (w - 20, 80), (200, 200, 200), 2)

                    if count >= FRAMES_PER_SIGN:
                        is_recording = False
                        append_data_to_csv(DATASET_FILE, current_label, collected_samples)
                        collected_samples = []
                        existing_signs = get_existing_classes(DATASET_FILE)
                        status_message = f"SUCCESS! Saved 100 frames for '{current_label}'"
                else:
                    cv2.putText(
                        frame, "HOLD HAND IN CAMERA VIEW!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA
                    )

            # --- State D: Idle Standby ---
            else:
                # Top header bar
                cv2.rectangle(frame, (0, 0), (w, 85), (20, 20, 20), -1)
                cv2.putText(
                    frame, f"Target Sign: {current_label}", (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
                )
                saved_str = ", ".join(existing_signs) if existing_signs else "None"
                cv2.putText(
                    frame, f"Saved in CSV: [{saved_str}]", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA
                )

                # Hand status badge
                if hand_in_view:
                    cv2.putText(frame, "Hand: Ready", (w - 180, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Hand: Not Detected", (w - 240, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 1, cv2.LINE_AA)

                # Footer instructions
                cv2.rectangle(frame, (0, h - 45), (w, h), (15, 15, 15), -1)
                cv2.putText(
                    frame, "[SPACE] Record 100 frames  |  [N] Change Sign  |  [Q] Exit",
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA
                )

            cv2.imshow("Sign Language Data Collector", frame)
            raw_key = cv2.waitKey(1)
            key = raw_key & 0xFF

            # Keystroke handling
            if is_typing_label:
                if key in [13, 10]:  # Enter
                    if typed_buffer.strip():
                        current_label = typed_buffer.strip().upper()
                        status_message = f"Active sign changed to: {current_label}"
                    is_typing_label = False
                    typed_buffer = ""
                elif key == 27:  # ESC
                    is_typing_label = False
                    typed_buffer = ""
                elif key == 8:  # Backspace
                    typed_buffer = typed_buffer[:-1]
                elif 32 <= key <= 126:  # Printable character
                    typed_buffer += chr(key).upper()
            else:
                if key in [ord('q'), 27]:
                    break
                elif key in [ord(' '), ord('s')] and not is_recording and not countdown_active:
                    countdown_active = True
                    countdown_start_time = time.time()
                elif key == ord('n') and not is_recording:
                    is_typing_label = True
                    typed_buffer = ""

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam released cleanly.")


if __name__ == "__main__":
    main()
