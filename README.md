# 🤟 AI Sign Language to Speech Desktop Pipeline

An end-to-end, lightweight, and real-time assistive technology desktop application that bridges communication barriers by translating American Sign Language (ASL) and custom hand gestures into synthesized speech in real time.

Built with **Python**, **OpenCV**, **MediaPipe Hands**, **Scikit-Learn**, and **Windows SAPI / pyttsx3**.

---

## 📌 Executive Summary & Problem Statement

Millions of individuals with speech and hearing impairments rely on sign language as their primary mode of communication. However, the majority of the general public cannot interpret sign language, creating communication hurdles in daily interactions, healthcare, and education.

This project delivers a **standalone, low-latency, real-time desktop pipeline** that requires no specialized hardware—operating smoothly on standard laptop webcams at 30 FPS.

---

## ⚡ Core Technical Features

1. **Dual-Hand Simultaneous Tracking**:
   * Tracks up to 2 hands (Left & Right) concurrently with 21 3D skeletal landmarks per hand ($2 \times 21 = 42$ spatial points).
   * Floating HUD badges track individual hand confidences in real time.

2. **Translation & Scale Invariance (Spatial Normalization)**:
   * **Translation Invariance**: All 21 landmarks are normalized relative to the wrist (landmark 0), ensuring the gesture is classified identically regardless of where the hand is in the camera frame:
     $$\vec{P}'_i = P_i - P_0$$
   * **Scale/Distance Invariance**: Normalized by maximum coordinate span, allowing users to stand close to or far from the webcam without distorting classification accuracy:
     $$\vec{P}'' = \frac{\vec{P}'}{\max(|\vec{P}'|)}$$

3. **High-Speed Guided Collection with 3D Data Augmentation**:
   * Includes `script1_fast_collector.py` with hands-free queue management.
   * Synthesizes 3D in-plane micro-rotations ($\pm 5^\circ$) and distance scaling to multiply 1 second of capture into **150 robust training frames per sign**.

4. **Non-Blocking Asynchronous Speech Engine**:
   * Uses native asynchronous Windows speech synthesis (`SAPI.SpVoice` with `SVSFlagsAsync`).
   * Eliminates video lag, frame drops, and audio deadlocks.

5. **Sentence Builder & Hands-Free Auto-Speech**:
   * Words accumulate in real time into an on-screen sentence bar (e.g. `Sentence: HELLO YES`).
   * **Hands-Free Mode**: Lowering hands for 1.2 seconds automatically speaks the complete sentence without touching the keyboard.
   * **Manual Mode**: Pressing `[SPACE]` or `[ENTER]` triggers instant full-phrase speech.

---

## 🏗️ System Architecture

```
[ Webcam Feed (30 FPS) ]
          │
          ▼
[ MediaPipe Hands (Dual-Hand Detection) ]
          │ (21 3D Landmarks per Hand)
          ▼
[ Spatial & Scale Normalization ]
   * Wrist-Relative Centering: P_i - P_0
   * Coordinate Max-Span Normalization: P / max(|P|)
          │ (63D Invariant Feature Vectors)
          ▼
[ Machine Learning Inference ]
   * Random Forest / High-Speed NumPy KNN
   * Confidence Thresholding (> 80%)
          │
          ▼
[ Sentence Accumulator & Temporal Debouncer ]
   * Stability Window (6-frame agreement)
   * Real-time UI HUD Overlay
          │
          ▼
[ Asynchronous Voice Synthesis (SAPI.SpVoice / pyttsx3) ]
   * Instant Word Output
   * Hands-Free Auto-Sentence Synthesis on Pause
```

---

## 📁 Repository Structure

```text
sign_language_pipeline/
├── script1_collect_data.py     # Interactive GUI-based single sign recorder
├── script1_fast_collector.py   # Hands-free batch collector with 3D data augmentation
├── script2_train_model.py      # Random Forest / NumPy KNN model trainer
├── script3_live_translator.py  # Live dual-hand translator with voice synthesis
├── requirements.txt            # Python dependencies
├── dataset.csv                 # (Generated) Normalized landmark dataset
├── model.pkl                   # (Generated) Serialized classifier bundle
└── .gitignore                  # Git exclusions
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/sign-language-translator.git
cd sign-language-translator
pip install -r requirements.txt
```

### 2. Record Custom Signs (Takes < 1 minute)
Launch the automated fast collector:
```bash
python script1_fast_collector.py
```
* Enter your desired words (e.g. `HELLO, YES, NO, THANKS, HELP`).
* Follow the on-screen countdowns. Hold each gesture for just 1 second.

### 3. Train the Classifier (Takes ~2 seconds)
```bash
python script2_train_model.py
```
* Generates train/test split, accuracy report, and saves `model.pkl`.

### 4. Run Live Translator
```bash
python script3_live_translator.py
```
* **Sign words** with one or both hands $\rightarrow$ words appear on screen.
* **Lower your hands** $\rightarrow$ computer automatically speaks the sentence!
* **Press `[SPACE]`** $\rightarrow$ manually speaks the full phrase on demand.
* **Press `[C]`** $\rightarrow$ clear sentence.
* **Press `[Q]`** $\rightarrow$ clean exit.

---

## 📈 Future Roadmap & Scalability

- [ ] **Dynamic Sign Recognition**: Integration of Temporal Convolutional Networks (TCN) or Bidirectional LSTMs for continuous motion signs.
- [ ] **Multi-Language Speech Synthesis**: Support for regional and international languages (e.g. Hindi, Spanish, French, Japanese).
- [ ] **Mobile & Edge Deployment**: Exporting trained models via ONNX and TensorFlow Lite for edge mobile devices (Android/iOS) and Raspberry Pi.
- [ ] **Facial Expression & Emotional Context**: Incorporating MediaPipe Face Mesh to capture grammatical facial expressions essential to sign language dialects.

---

## 📜 License
This project is licensed under the MIT License - open for educational and research use.
