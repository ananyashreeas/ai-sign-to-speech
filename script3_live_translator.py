"""
script3_live_translator.py
--------------------------
Real-time Sign Language to Speech translation desktop application with:
1. SIMULTANEOUS DUAL-HAND TRACKING (num_hands = 2)
2. NATIVE ASYNCHRONOUS SPEECH (SAPI.SpVoice - Zero Freezes / Hangs)
3. IN-APP ON-SCREEN SIGN CREATOR ([N] Button):
   - Press [N] directly on this webcam window anytime!
   - Type the new word directly on screen and hit ENTER.
   - Hold your gesture for 1 second.
   - The AI automatically learns, retrains in the background, and immediately
     starts translating your new sign without ever closing or restarting!
4. SENTENCE BUILDER & HANDS-FREE AUTO-SPEAKING on pause/lower hand.
"""

import os
# Suppress Google/MediaPipe C++ telemetry and logging BEFORE importing mediapipe
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import csv
import time
import pickle
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from script2_train_model import (
    NumpyKNNClassifier,
    load_dataset_csv,
    split_data,
    build_and_train_model,
    save_model,
    DATASET_FILE
)

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
MODEL_FILE = "model.pkl"
CONFIDENCE_THRESHOLD = 0.80       # 80% minimum confidence threshold
CONSECUTIVE_FRAMES_TRIGGER = 6    # Quick responsiveness: 6 frames of agreement
SPEECH_COOLDOWN_SECONDS = 1.8     # Minimum seconds between repeating the same held sign
AUTO_SPEAK_PAUSE_SECONDS = 1.2    # Seconds after lowering hands before speaking full sentence
MAX_NUM_HANDS = 2                 # Support up to 2 hands simultaneously
NEW_SIGN_RAW_FRAMES = 30          # Only 30 frames (~1 sec) needed for new signs
WEBCAM_INDEX = 0
TARGET_FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_task_model():
    if not os.path.exists(TASK_MODEL_PATH):
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
                frame_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        else:
            self.drawing_utils.draw_landmarks(frame_bgr, hand_landmarks, self.connections)

    def close(self):
        if self.mode == "solutions":
            self.detector.close()


class NativeSpeechEngine:
    """Non-blocking Windows Speech Synthesizer."""

    def __init__(self, rate=1):
        self.use_win32 = False
        try:
            import win32com.client
            self.speaker = win32com.client.Dispatch("SAPI.SpVoice")
            self.speaker.Rate = rate
            self.use_win32 = True
        except Exception:
            import pyttsx3
            self.pyttsx3_engine = pyttsx3.init()
            self.pyttsx3_engine.setProperty("rate", 160)

    def speak(self, text: str, interrupt: bool = False):
        if not text or not text.strip():
            return
        clean_text = text.strip()
        if self.use_win32:
            try:
                flags = (1 | 2) if interrupt else 1
                self.speaker.Speak(clean_text, flags)
            except Exception:
                pass
        else:
            try:
                self.pyttsx3_engine.say(clean_text)
                self.pyttsx3_engine.runAndWait()
            except Exception:
                pass

    def stop(self):
        if self.use_win32:
            try:
                self.speaker.Speak("", 2)
            except Exception:
                pass


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


def augment_landmark_vector(vector_63: np.ndarray) -> list:
    """Generates 5 3D geometric variations for rapid learning."""
    augmented = [vector_63]
    coords = vector_63.reshape(21, 3)
    for angle in [-0.08, 0.08]:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
        rotated = np.dot(coords, rot_matrix)
        augmented.append(rotated.flatten())
    augmented.append((coords * 1.05).flatten())
    augmented.append((coords * 0.95).flatten())
    return augmented


def append_data_to_csv(filepath: str, label: str, samples: list):
    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for s in samples:
            writer.writerow([label] + s.tolist())


def retrain_model_pipeline(filepath: str):
    """Loads CSV, fits model, saves model.pkl, and returns updated model + classes."""
    X, y, classes = load_dataset_csv(filepath)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model, _ = build_and_train_model(X_train, y_train)
    save_model(model, MODEL_FILE)
    return model, list(model.classes_)


def load_model(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Trained model '{filepath}' not found! Please run 'script2_train_model.py' first."
        )

    with open(filepath, "rb") as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        classes = bundle.get("classes", list(model.classes_))
    else:
        model = bundle
        classes = list(model.classes_)

    return model, classes


def draw_ui_overlay(frame, hand_predictions, sentence_words, speech_banner, auto_countdown):
    """Renders status HUD, active dual-hand detections, and running sentence bar."""
    h, w, _ = frame.shape

    # --- Top Banner (Status Bar) ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    num_hands = len(hand_predictions)
    if num_hands == 0:
        cv2.putText(frame, "Hands: Lowered / Paused", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 180, 255), 2, cv2.LINE_AA)
        if auto_countdown is not None and auto_countdown > 0:
            cv2.putText(frame, f"Auto-speaking sentence in {auto_countdown:.1f}s...", (20, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Raise hand(s) to sign | [N] to Teach New Sign", (20, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"Hands Active: {num_hands} detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

        summary_parts = []
        for i, (pred, conf) in enumerate(hand_predictions):
            pct = int(conf * 100)
            if conf >= CONFIDENCE_THRESHOLD:
                summary_parts.append(f"H{i+1}: {pred.replace('_', ' ')} ({pct}%)")
            else:
                summary_parts.append(f"H{i+1}: ? ({pct}%)")
        cv2.putText(frame, " | ".join(summary_parts), (20, 58),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    # Active Speech Banner on Top Right
    if speech_banner:
        banner_w = min(320, len(speech_banner) * 12 + 40)
        cv2.rectangle(frame, (w - banner_w - 15, 10), (w - 15, 60), (0, 140, 0), -1)
        cv2.putText(frame, speech_banner, (w - banner_w - 5, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Bottom Banner (Sentence Builder Bar) ---
    sentence_overlay = frame.copy()
    cv2.rectangle(sentence_overlay, (0, h - 85), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(sentence_overlay, 0.85, frame, 0.15, 0, frame)

    sentence_str = " ".join([w.replace("_", " ") for w in sentence_words]) if sentence_words else "(Empty - hold gestures to add words)"
    color = (0, 255, 255) if sentence_words else (140, 140, 140)
    cv2.putText(frame, f"Sentence: {sentence_str}", (15, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    controls_text = "[SPACE] Speak Now | [N] Add New Sign | Lower Hands: Auto-Speak | [C] Clear | [Q] Quit"
    cv2.putText(frame, controls_text, (15, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1, cv2.LINE_AA)


def main():
    try:
        model, classes = load_model(MODEL_FILE)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    tts = NativeSpeechEngine(rate=1)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Failed to open webcam index {WEBCAM_INDEX}.")
        return

    WINDOW_NAME = "Sign Language to Speech Translator"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 800, 600)

    detector = HandDetector(max_hands=MAX_NUM_HANDS)

    # Per-hand stability tracking
    hand_candidates = [None, None]
    hand_consecutive_hits = [0, 0]
    hand_last_spoken_sign = [None, None]
    hand_last_spoken_time = [0.0, 0.0]

    # Sentence & timing states
    sentence_words = []
    last_hand_seen_time = time.time()
    speech_banner = None
    speech_banner_time = 0.0

    # In-App Sign Creator states
    is_typing_new_sign = False
    typed_buffer = ""
    is_recording_new_sign = False
    new_sign_target = ""
    new_sign_samples = []
    new_sign_countdown_start = 0.0

    print("\n" + "=" * 65)
    print(" LIVE SIGN TRANSLATOR & IN-APP SIGN CREATOR")
    print("=" * 65)
    print(f"Supported signs : {', '.join(classes)}")
    print(f"Max hands       : {MAX_NUM_HANDS} simultaneous")
    print("Features:")
    print("  - Sign words with one or both hands -> SPEAKS IMMEDIATELY!")
    print("  - Lower both hands                  -> AUTO-SPEAKS the full sentence!")
    print("  - Press [N] ON CAMERA SCREEN        -> TEACH & ADD A NEW SIGN LIVE!")
    print("  - Press [SPACE] / [S]               -> Speak sentence now")
    print("Controls:")
    print("  [N] Teach New Sign | [SPACE] Speak | [C] Clear | [Q] Quit")
    print("=" * 65 + "\n")

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
            current_hand_predictions = []
            auto_countdown = None

            # --- MODE 1: On-Screen Text Input Modal (Triggered by [N]) ---
            if is_typing_new_sign:
                modal = frame.copy()
                cv2.rectangle(modal, (40, h // 2 - 90), (w - 40, h // 2 + 75), (25, 25, 25), -1)
                cv2.addWeighted(modal, 0.88, frame, 0.12, 0, frame)
                cv2.rectangle(frame, (40, h // 2 - 90), (w - 40, h // 2 + 75), (0, 220, 255), 2)

                cv2.putText(frame, "TEACH NEW SIGN: TYPE WORD & PRESS ENTER",
                            (60, h // 2 - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)

                cursor = "_" if int(now * 2) % 2 == 0 else ""
                cv2.putText(frame, f"> {typed_buffer}{cursor}",
                            (60, h // 2 + 5), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(frame, "[ENTER] Start 1-Sec Recording  |  [ESC] Cancel",
                            (60, h // 2 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1, cv2.LINE_AA)

            # --- MODE 2: Recording New Sign Pose (3-Sec Countdown & 1-Sec Capture) ---
            elif is_recording_new_sign:
                elapsed = now - new_sign_countdown_start
                remaining = int(3.5 - elapsed)

                if hand_in_view:
                    for hlms in hands_detected:
                        detector.draw_landmarks(frame, hlms)

                if remaining > 0:
                    # 3-Second Countdown
                    cv2.putText(frame, f"GET READY FOR '{new_sign_target}': {remaining}",
                                (w // 2 - 230, h // 2), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 140, 255), 3, cv2.LINE_AA)
                else:
                    # Capture 30 frames
                    if hand_in_view:
                        primary_hand = hands_detected[0]
                        norm_vec = normalize_landmarks(primary_hand)
                        new_sign_samples.append(norm_vec)

                        count = len(new_sign_samples)
                        cv2.putText(frame, f"HOLD GESTURE '{new_sign_target}': {count}/{NEW_SIGN_RAW_FRAMES}",
                                    (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 0), 2)

                        bar_w = int((count / NEW_SIGN_RAW_FRAMES) * (w - 40))
                        cv2.rectangle(frame, (20, 65), (20 + bar_w, 80), (0, 255, 0), -1)

                        if count >= NEW_SIGN_RAW_FRAMES:
                            # 3D Geometric Augmentation & Retraining
                            all_augmented = []
                            for vec in new_sign_samples:
                                all_augmented.extend(augment_landmark_vector(vec))

                            append_data_to_csv(DATASET_FILE, new_sign_target, all_augmented)
                            print(f"[SUCCESS] Appended {len(all_augmented)} samples for '{new_sign_target}'. Retraining model...")

                            # Retrain in-memory
                            model, classes = retrain_model_pipeline(DATASET_FILE)
                            print(f"[RE-TRAINED] Model updated in-memory! Supported: {classes}")

                            tts.speak(f"Learned new sign {new_sign_target.replace('_', ' ')}")
                            speech_banner = f"Learned: {new_sign_target}"
                            speech_banner_time = now

                            is_recording_new_sign = False
                            new_sign_samples = []
                    else:
                        cv2.putText(frame, f"HOLD HAND IN VIEW FOR '{new_sign_target}'!",
                                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # --- MODE 3: Normal Real-Time Translation ---
            else:
                if hand_in_view:
                    last_hand_seen_time = now

                    for i, hand_lms in enumerate(hands_detected[:MAX_NUM_HANDS]):
                        detector.draw_landmarks(frame, hand_lms)

                        raw_wrist = hand_lms[0] if not hasattr(hand_lms, "landmark") else hand_lms.landmark[0]
                        wx = int(raw_wrist.x * w)
                        wy = int(raw_wrist.y * h)

                        features = normalize_landmarks(hand_lms).reshape(1, -1)
                        probabilities = model.predict_proba(features)[0]
                        best_class_idx = np.argmax(probabilities)
                        conf = float(probabilities[best_class_idx])
                        pred = str(model.classes_[best_class_idx])

                        current_hand_predictions.append((pred, conf))

                        if conf >= CONFIDENCE_THRESHOLD:
                            pct = int(conf * 100)
                            clean_label = pred.replace("_", " ")
                            badge_text = f"Hand {i+1}: {clean_label} ({pct}%)"
                            badge_x = max(10, min(w - 200, wx - 60))
                            badge_y = max(40, wy - 25)

                            cv2.rectangle(frame, (badge_x - 5, badge_y - 20), (badge_x + 195, badge_y + 5), (0, 150, 0), -1)
                            cv2.putText(frame, badge_text, (badge_x, badge_y - 5),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                            if pred == hand_candidates[i]:
                                hand_consecutive_hits[i] += 1
                            else:
                                hand_candidates[i] = pred
                                hand_consecutive_hits[i] = 1

                            if hand_consecutive_hits[i] >= CONSECUTIVE_FRAMES_TRIGGER:
                                is_new_sign = (pred != hand_last_spoken_sign[i])
                                cooldown_passed = (now - hand_last_spoken_time[i] > SPEECH_COOLDOWN_SECONDS)

                                if is_new_sign or cooldown_passed:
                                    spoken_text = pred.replace("_", " ")
                                    print(f"[SPEAK] Hand {i+1}: '{spoken_text}' ({pct}%)")
                                    tts.speak(spoken_text, interrupt=False)
                                    hand_last_spoken_sign[i] = pred
                                    hand_last_spoken_time[i] = now
                                    speech_banner = f"Spoke: {spoken_text}"
                                    speech_banner_time = now

                                    if not sentence_words or sentence_words[-1] != pred:
                                        sentence_words.append(pred)
                        else:
                            hand_candidates[i] = None
                            hand_consecutive_hits[i] = 0

                    if len(hands_detected) == 1:
                        hand_candidates[1] = None
                        hand_consecutive_hits[1] = 0
                        hand_last_spoken_sign[1] = None

                else:
                    hand_candidates = [None, None]
                    hand_consecutive_hits = [0, 0]
                    hand_last_spoken_sign = [None, None]
                    hand_last_spoken_time = [0.0, 0.0]

                    if sentence_words:
                        pause_duration = now - last_hand_seen_time
                        remaining_time = AUTO_SPEAK_PAUSE_SECONDS - pause_duration
                        if remaining_time > 0:
                            auto_countdown = remaining_time
                        else:
                            full_sentence = " ".join([w.replace("_", " ") for w in sentence_words])
                            print(f"[AUTO-SPEAK] Full sentence: '{full_sentence}'")
                            tts.speak(full_sentence, interrupt=True)
                            speech_banner = f"Sentence: {full_sentence}"
                            speech_banner_time = now
                            sentence_words = []
                            last_hand_seen_time = now

                if speech_banner and (now - speech_banner_time > 2.0):
                    speech_banner = None

                draw_ui_overlay(
                    frame,
                    hand_predictions=current_hand_predictions,
                    sentence_words=sentence_words,
                    speech_banner=speech_banner,
                    auto_countdown=auto_countdown
                )

            cv2.imshow(WINDOW_NAME, frame)
            raw_key = cv2.waitKey(1)
            key = raw_key & 0xFF

            # Key Handling
            if is_typing_new_sign:
                if key in [13, 10]:  # ENTER: Confirm sign name and start capture
                    clean_name = typed_buffer.strip().replace(" ", "_").upper()
                    if clean_name:
                        new_sign_target = clean_name
                        is_typing_new_sign = False
                        is_recording_new_sign = True
                        new_sign_samples = []
                        new_sign_countdown_start = time.time()
                    else:
                        is_typing_new_sign = False
                elif key == 27:  # ESC: Cancel
                    is_typing_new_sign = False
                    typed_buffer = ""
                elif key == 8:  # BACKSPACE
                    typed_buffer = typed_buffer[:-1]
                elif 32 <= key <= 126:  # Printable character
                    typed_buffer += chr(key).upper()

            elif is_recording_new_sign:
                if key == 27:  # ESC: Cancel recording
                    is_recording_new_sign = False
                    new_sign_samples = []

            else:
                if key in [ord('q'), 27]:
                    print("[INFO] Exiting translator.")
                    break
                elif key in [ord('n'), ord('N')]:
                    # Trigger In-App Sign Creator
                    is_typing_new_sign = True
                    typed_buffer = ""
                elif key in [32, ord(' '), ord('s'), ord('S'), 13]:
                    if sentence_words:
                        full_sentence = " ".join([w.replace("_", " ") for w in sentence_words])
                        print(f"[MANUAL SPEAK] Speaking sentence: '{full_sentence}'")
                        tts.speak(full_sentence, interrupt=True)
                        speech_banner = f"Sentence: {full_sentence}"
                        speech_banner_time = now
                        sentence_words = []
                elif key in [ord('c'), ord('C')]:
                    sentence_words = []
                    speech_banner = "Cleared"
                    speech_banner_time = now
                    print("[SENTENCE] Sentence cleared.")
                elif key == 8:  # Backspace
                    if sentence_words:
                        popped = sentence_words.pop()
                        print(f"[SENTENCE] Removed '{popped}'.")

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        tts.stop()
        print("[INFO] Resources released cleanly.")


if __name__ == "__main__":
    main()
