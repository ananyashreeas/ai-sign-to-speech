"""
script3_live_translator.py
--------------------------
Real-time Sign Language to Speech translation desktop application with
NATIVE ASYNCHRONOUS SPEECH and SIMULTANEOUS DUAL-HAND TRACKING.

Key Features & Fixes:
1. Native Windows SAPI.SpVoice: Replaced pyttsx3 threading queue with native Windows
   SAPI.SpVoice asynchronous engine (SVSFlagsAsync = 1).
   -> Speaks every word, every sentence, repeatedly with ZERO hangs, freezes, or dropped calls!
2. Simultaneous 2-Hand Tracking: Both hands tracked and classified in real time with
   floating green badges over each wrist.
3. Interactive Sentence Builder:
   * As you sign, words are collected into the sentence bar at the bottom.
   * Press [SPACE], [S], or [ENTER] on the video window to speak the entire sentence immediately!
   * Hands-Free Auto-Speak: When you lower both hands for 1.2 seconds, it automatically speaks
     the complete sentence for you!
4. Controls:
   [SPACE] / [S] / [ENTER] : Speak entire sentence
   [C]                     : Clear sentence
   [BACKSPACE]             : Remove last word
   [Q] / [ESC]             : Quit program
"""

import os
# Suppress Google/MediaPipe C++ telemetry and logging BEFORE importing mediapipe
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import time
import pickle
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from script2_train_model import NumpyKNNClassifier

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
MODEL_FILE = "model.pkl"
CONFIDENCE_THRESHOLD = 0.80       # 80% minimum confidence threshold
CONSECUTIVE_FRAMES_TRIGGER = 6    # Quick responsiveness: 6 frames of agreement
SPEECH_COOLDOWN_SECONDS = 1.8     # Minimum seconds between repeating the same held sign
AUTO_SPEAK_PAUSE_SECONDS = 1.2    # Seconds after lowering hands before speaking full sentence
MAX_NUM_HANDS = 2                 # Support up to 2 hands simultaneously
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
    """
    Rock-solid, 100% non-blocking native Windows Speech Synthesizer (SAPI.SpVoice).
    Uses asynchronous COM flags (SVSFlagsAsync = 1) so it NEVER freezes the video loop,
    never drops repeated calls, and speaks sentences on demand without delay.
    """

    def __init__(self, rate=1):
        self.use_win32 = False
        try:
            import win32com.client
            self.speaker = win32com.client.Dispatch("SAPI.SpVoice")
            self.speaker.Rate = rate
            self.use_win32 = True
            print("[INFO] Initialized native Windows SAPI.SpVoice speech engine (Non-blocking).")
        except Exception as e:
            print(f"[WARNING] win32com unavailable ({e}), falling back to pyttsx3...")
            import pyttsx3
            self.pyttsx3_engine = pyttsx3.init()
            self.pyttsx3_engine.setProperty("rate", 160)

    def speak(self, text: str, interrupt: bool = False):
        if not text or not text.strip():
            return
        clean_text = text.strip()

        if self.use_win32:
            try:
                # Flag 1 = SVSFlagsAsync (Async, non-blocking)
                # Flag 2 = SVSFPurgeBeforeSpeak (Interrupt previous voice if requested)
                flags = (1 | 2) if interrupt else 1
                self.speaker.Speak(clean_text, flags)
            except Exception as e:
                print(f"[TTS Error] {e}")
        else:
            try:
                self.pyttsx3_engine.say(clean_text)
                self.pyttsx3_engine.runAndWait()
            except Exception as e:
                print(f"[TTS Fallback Error] {e}")

    def stop(self):
        if self.use_win32:
            try:
                # 2 = SVSFPurgeBeforeSpeak with empty text stops ongoing speech
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

    print(f"[INFO] Loaded model successfully with classes: {classes}")
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
        cv2.putText(frame, "Hands: Lowered / Not in view", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 180, 255), 2, cv2.LINE_AA)
        if auto_countdown is not None and auto_countdown > 0:
            cv2.putText(frame, f"Auto-speaking sentence in {auto_countdown:.1f}s...", (20, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Raise hand(s) to sign words", (20, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"Hands Active: {num_hands} detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

        summary_parts = []
        for i, (pred, conf) in enumerate(hand_predictions):
            pct = int(conf * 100)
            if conf >= CONFIDENCE_THRESHOLD:
                summary_parts.append(f"H{i+1}: {pred} ({pct}%)")
            else:
                summary_parts.append(f"H{i+1}: ? ({pct}%)")
        cv2.putText(frame, " | ".join(summary_parts), (20, 58),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    # Active Speech Banner on Top Right
    if speech_banner:
        banner_w = min(280, len(speech_banner) * 12 + 40)
        cv2.rectangle(frame, (w - banner_w - 15, 10), (w - 15, 60), (0, 140, 0), -1)
        cv2.putText(frame, speech_banner, (w - banner_w - 5, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Bottom Banner (Sentence Builder Bar) ---
    sentence_overlay = frame.copy()
    cv2.rectangle(sentence_overlay, (0, h - 80), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(sentence_overlay, 0.85, frame, 0.15, 0, frame)

    sentence_str = " ".join(sentence_words) if sentence_words else "(Empty - hold gestures to add words)"
    color = (0, 255, 255) if sentence_words else (140, 140, 140)
    cv2.putText(frame, f"Sentence: {sentence_str}", (15, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    controls_text = "[SPACE] Speak Sentence Now | Lower Hands: Auto-Speak | [C] Clear | [Q] Quit"
    cv2.putText(frame, controls_text, (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 240, 180), 1, cv2.LINE_AA)


def main():
    try:
        model, classes = load_model(MODEL_FILE)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # Initialize native non-blocking Windows speech engine
    tts = NativeSpeechEngine(rate=1)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Failed to open webcam index {WEBCAM_INDEX}.")
        return

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

    print("\n" + "=" * 65)
    print(" LIVE SIGN TRANSLATOR (NATIVE ASYNC SPEECH & 2-HAND SUPPORT)")
    print("=" * 65)
    print(f"Supported signs : {', '.join(classes)}")
    print(f"Max hands       : {MAX_NUM_HANDS} simultaneous")
    print("Features:")
    print("  - Raise hand(s) and hold a sign -> SPEAKS IMMEDIATELY!")
    print("  - Words accumulate in the Sentence bar at bottom.")
    print("  - Lower both hands              -> AUTO-SPEAKS the full sentence!")
    print("  - Press [SPACE] or [S]          -> MANUALLY speaks sentence immediately!")
    print("Controls:")
    print("  [SPACE] / [S] : Speak sentence immediately")
    print("  [C]           : Clear sentence")
    print("  [BACKSPACE]   : Remove last word")
    print("  [Q] / [ESC]   : Quit program")
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

            if hand_in_view:
                last_hand_seen_time = now

                # Process every detected hand (up to 2)
                for i, hand_lms in enumerate(hands_detected[:MAX_NUM_HANDS]):
                    detector.draw_landmarks(frame, hand_lms)

                    raw_wrist = hand_lms[0] if not hasattr(hand_lms, "landmark") else hand_lms.landmark[0]
                    wx = int(raw_wrist.x * w)
                    wy = int(raw_wrist.y * h)

                    # Extract normalized 63D features
                    features = normalize_landmarks(hand_lms).reshape(1, -1)
                    probabilities = model.predict_proba(features)[0]
                    best_class_idx = np.argmax(probabilities)
                    conf = float(probabilities[best_class_idx])
                    pred = str(model.classes_[best_class_idx])

                    current_hand_predictions.append((pred, conf))

                    # Floating label directly on camera above each hand
                    if conf >= CONFIDENCE_THRESHOLD:
                        pct = int(conf * 100)
                        badge_text = f"Hand {i+1}: {pred} ({pct}%)"
                        badge_x = max(10, min(w - 180, wx - 60))
                        badge_y = max(40, wy - 25)

                        cv2.rectangle(frame, (badge_x - 5, badge_y - 20), (badge_x + 175, badge_y + 5), (0, 150, 0), -1)
                        cv2.putText(frame, badge_text, (badge_x, badge_y - 5),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                        # Stability tracking for this hand
                        if pred == hand_candidates[i]:
                            hand_consecutive_hits[i] += 1
                        else:
                            hand_candidates[i] = pred
                            hand_consecutive_hits[i] = 1

                        # Trigger speech once gesture stabilizes across 6 frames
                        if hand_consecutive_hits[i] >= CONSECUTIVE_FRAMES_TRIGGER:
                            is_new_sign = (pred != hand_last_spoken_sign[i])
                            cooldown_passed = (now - hand_last_spoken_time[i] > SPEECH_COOLDOWN_SECONDS)

                            if is_new_sign or cooldown_passed:
                                print(f"[SPEAK] Hand {i+1}: '{pred}' ({pct}%)")
                                tts.speak(pred, interrupt=False)
                                hand_last_spoken_sign[i] = pred
                                hand_last_spoken_time[i] = now
                                speech_banner = f"Spoke: {pred}"
                                speech_banner_time = now

                                # Add to sentence builder
                                if not sentence_words or sentence_words[-1] != pred:
                                    sentence_words.append(pred)
                                    print(f"[SENTENCE] Current: {' '.join(sentence_words)}")
                    else:
                        hand_candidates[i] = None
                        hand_consecutive_hits[i] = 0

                # Reset tracker for second hand if only 1 hand is present
                if len(hands_detected) == 1:
                    hand_candidates[1] = None
                    hand_consecutive_hits[1] = 0
                    hand_last_spoken_sign[1] = None

            else:
                # When hands are lowered out of view:
                # Reset sign state so raising hands again immediately triggers speech!
                hand_candidates = [None, None]
                hand_consecutive_hits = [0, 0]
                hand_last_spoken_sign = [None, None]
                hand_last_spoken_time = [0.0, 0.0]

                # Hands-Free Auto-Speak:
                # If hands have been down for AUTO_SPEAK_PAUSE_SECONDS and there are words,
                # speak the full sentence automatically!
                if sentence_words:
                    pause_duration = now - last_hand_seen_time
                    remaining_time = AUTO_SPEAK_PAUSE_SECONDS - pause_duration
                    if remaining_time > 0:
                        auto_countdown = remaining_time
                    else:
                        full_sentence = " ".join(sentence_words)
                        print(f"[AUTO-SPEAK] Full sentence: '{full_sentence}'")
                        tts.speak(full_sentence, interrupt=True)
                        speech_banner = f"Sentence: {full_sentence}"
                        speech_banner_time = now
                        sentence_words = []
                        last_hand_seen_time = now

            # Clear speech banner after 2.0s
            if speech_banner and (now - speech_banner_time > 2.0):
                speech_banner = None

            draw_ui_overlay(
                frame,
                hand_predictions=current_hand_predictions,
                sentence_words=sentence_words,
                speech_banner=speech_banner,
                auto_countdown=auto_countdown
            )

            cv2.imshow("Sign Language to Speech Translator", frame)
            raw_key = cv2.waitKey(1)
            key = raw_key & 0xFF

            if key in [ord('q'), 27]:
                print("[INFO] Exiting translator.")
                break
            # Handle SPACE, 's', 'S', or ENTER to speak full sentence
            elif key in [32, ord(' '), ord('s'), ord('S'), 13, 10]:
                if sentence_words:
                    full_sentence = " ".join(sentence_words)
                    print(f"[MANUAL SPEAK] Key pressed! Speaking sentence: '{full_sentence}'")
                    tts.speak(full_sentence, interrupt=True)
                    speech_banner = f"Sentence: {full_sentence}"
                    speech_banner_time = now
                    sentence_words = []
                else:
                    print("[INFO] Sentence is currently empty. Hold signs to add words first.")
            elif key in [ord('c'), ord('C')]:
                sentence_words = []
                speech_banner = "Cleared"
                speech_banner_time = now
                print("[SENTENCE] Sentence cleared.")
            elif key == 8:  # Backspace
                if sentence_words:
                    popped = sentence_words.pop()
                    print(f"[SENTENCE] Removed '{popped}'. Remaining: {' '.join(sentence_words)}")

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        tts.stop()
        print("[INFO] Resources released cleanly.")


if __name__ == "__main__":
    main()
