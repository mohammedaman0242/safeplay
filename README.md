@'
# SafePlay — Volleyball Motion Risk Analysis

SafePlay is a computer vision and deep learning system for analyzing volleyball movements and identifying potentially risky motion patterns from video.

The system processes volleyball movement videos using pose estimation, motion-feature extraction, temporal sequence modeling, and LSTM-based classification. It can generate frame-level and window-level risk probabilities together with visual reports.

## Project Overview

SafePlay follows an end-to-end video analysis pipeline:

Video
→ MediaPipe Pose
→ 17 body landmarks
→ 32 motion features
→ Temporal windows
→ Action-specific normalization
→ LSTM classification
→ Risk probabilities
→ Visualized results

The project supports three volleyball movement categories:

- Attack
- Block
- Defence

## Key Features

- Human pose extraction using MediaPipe Pose
- 32-dimensional motion feature representation
- Joint-angle and body-movement features
- Temporal sequence modeling using LSTM
- Action-specific feature normalization
- Risk probability estimation
- Frame-level and window-level analysis
- Annotated output videos
- Risk plots and CSV reports
- Streamlit-based interface for video analysis
- GAN-based sequence augmentation components

## Feature Representation

SafePlay uses 32 motion features derived from 17 pose landmarks.

The feature representation includes:

- Joint angles
- Vertical displacement
- Body proportions and distances
- Angular velocity features
- Torso movement features

The temporal model processes sequences of 30 frames.

## Model

The main classification model is an LSTM-based binary classifier.

The existing trained model uses:

- Sequence length: 30 frames
- Feature dimension: 32
- Hidden size: 128
- Binary classification:
  - SAFE
  - UNSAFE

Action-specific scalers are provided for:

- Attack
- Block
- Defence

## Demo Result

A real inference run is included in:

`results/demo_block/`

The demonstrated block sequence produced:

- Risk level: Moderate Risk
- Mean unsafe probability: 0.372956
- Maximum unsafe probability: 0.648379
- Windows above 0.5 probability: 12.50%

The result represents a demonstration inference on one video and should not be interpreted as a model accuracy or generalization metric.

## Project Structure

```text
safeplay/
│
├── app/
│   └── app_streamlit.py
│
├── assets/
│
├── configs/
│
├── models/
│   ├── attack_scaler.pkl
│   ├── block_scaler.pkl
│   ├── defence_scaler.pkl
│   ├── ae_threshold.npz
│   ├── lstm_autoencoder.pth
│   ├── lstm_autoencoder_best.pth
│   └── lstm_classifier_best.pth
│
├── results/
│   └── demo_block/
│
├── src/
│   ├── augmentation/
│   │   ├── augment.py
│   │   └── train_seq_gan.py
│   │
│   ├── inference/
│   │   └── full_report_from_video.py
│   │
│   ├── preprocessing/
│   │   ├── compute_features_and_smooth.py
│   │   ├── extract_keypoints.py
│   │   ├── features.py
│   │   ├── make_windows.py
│   │   └── normalize.py
│   │
│   └── training/
│       └── train_lstm_classifier.py
│
├── .gitignore
├── requirements.txt
└── README.md