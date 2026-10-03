#!/usr/bin/env python3
"""
full_report_from_video.py

Usage examples:
  # Process existing video (must pass --action)
  python full_report_from_video.py --video sample_videos/block/B1.mp4 --action block --out_dir reports

  # Capture 8 seconds from webcam then process (must pass --action)
  python full_report_from_video.py --capture 8 --action attack --out_dir reports

  # Process pre-normalized windows NPZ (no --action needed)
  python full_report_from_video.py --windows_npz windows/attack_windows_norm.npz --out_dir reports

Outputs:
  <out_dir>/<action>_<video_basename>_annotated.mp4
  <out_dir>/<action>_<video_basename>_window_probs.csv
  <out_dir>/<action>_<video_basename>_frame_probs.csv
  <out_dir>/<action>_<video_basename>_risk_plot.png
  <out_dir>/<action>_<video_basename>_frame_risk_plot.png
  <out_dir>/<action>_<video_basename>_summary.txt
"""

import os
import sys
import argparse
import time
import math
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional smoothing
try:
    from scipy.ndimage import gaussian_filter1d
except Exception:
    gaussian_filter1d = None

# Mediapipe
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 30
WINDOW_STEP = 15
BATCH_SIZE = 64

# Default paths (relative to script)
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = BASE_DIR / "models" / "lstm_classifier_best.pth"

# ---------------- Model definition (must match training) ----------------
class LSTMClassifier(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B,T,F)
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        logits = self.fc(h_last)  # (B,1)
        return logits.squeeze(1)  # (B,)

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X.astype("float32")
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return torch.from_numpy(self.X[idx])

# ---------------- Utilities ----------------
def safe_makedirs(p: Path):
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

def find_scaler_for_action(action: str):
    # tries a few common locations
    candidates = [
        BASE_DIR / "windows" / f"{action}_scaler.pkl",
        BASE_DIR / "models" / f"{action}_scaler.pkl",
        BASE_DIR / f"{action}_scaler.pkl",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

# ---------------- Keypoint extraction ----------------
def extract_keypoints_from_video(video_path):
    """
    Returns:
      keypoints_seq: (N_frames, 33, 3) normalized x,y,z in [0,1] (z is relative)
      frames: list of BGR frames (numpy arrays)
      (h,w): frame dimensions
      fps: frames per second value
    If MediaPipe not available raises RuntimeError.
    """
    if not MP_AVAILABLE:
        raise RuntimeError("mediapipe is not installed / importable. Install mediapipe to extract keypoints.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    keypoints = []
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)

    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.copy())
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                arr = []
                for lm in res.pose_landmarks.landmark:
                    arr.append([float(lm.x), float(lm.y), float(lm.z if lm.z is not None else 0.0)])
                arr = np.array(arr, dtype=np.float32)  # 33,3
            else:
                # No detection: use zeros (0,0,0) which denotes 'missing' for later drawing
                arr = np.zeros((33,3), dtype=np.float32)
            keypoints.append(arr)
    cap.release()

    keypoints = np.stack(keypoints, axis=0) if keypoints else np.zeros((0,33,3), dtype=np.float32)
    return keypoints, frames, (h,w), fps

# ---------------- Feature extraction ----------------
def angle_3pts(a,b,c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosv = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(math.acos(cosv)))

def dist(a,b):
    a = np.array(a, dtype=np.float32); b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a-b))

def extract_features_from_keypoints_seq(keypoints_seq, img_h, img_w, fps):
    """
    Build 32-d feature vector per frame matching your earlier pipeline.

    Steps:
      - compute angle features: knees, hips, elbows, shoulders (8 angles)
      - ankle vertical (normalized), wrist vertical
      - torso bend, shoulder/hip widths, torso length, mid_hip_y
      - add vertical displacement (px) and normalized vertical displacement
      - add angular velocities for first 13 angles

    Returns:
      features: (N_frames, 32) float32
    """
    N = keypoints_seq.shape[0]
    rows = []
    for t in range(N):
        kp = keypoints_seq[t]  # (33,3), normalized coords 0..1

        def xy(idx):
            # If coords are 0 (no detection), return center-ish to avoid NaNs
            x = float(kp[idx,0]) * img_w
            y = float(kp[idx,1]) * img_h
            return (x,y)

        # key landmarks indices (mediapipe)
        L_SH = xy(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        R_SH = xy(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        L_EL = xy(mp_pose.PoseLandmark.LEFT_ELBOW.value)
        R_EL = xy(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        L_WR = xy(mp_pose.PoseLandmark.LEFT_WRIST.value)
        R_WR = xy(mp_pose.PoseLandmark.RIGHT_WRIST.value)
        L_HIP = xy(mp_pose.PoseLandmark.LEFT_HIP.value)
        R_HIP = xy(mp_pose.PoseLandmark.RIGHT_HIP.value)
        L_KNEE = xy(mp_pose.PoseLandmark.LEFT_KNEE.value)
        R_KNEE = xy(mp_pose.PoseLandmark.RIGHT_KNEE.value)
        L_ANK = xy(mp_pose.PoseLandmark.LEFT_ANKLE.value)
        R_ANK = xy(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

        mid_sh = ((L_SH[0]+R_SH[0])/2.0, (L_SH[1]+R_SH[1])/2.0)
        mid_hip = ((L_HIP[0]+R_HIP[0])/2.0, (L_HIP[1]+R_HIP[1])/2.0)

        left_knee = angle_3pts(L_HIP, L_KNEE, L_ANK)
        right_knee = angle_3pts(R_HIP, R_KNEE, R_ANK)
        left_hip = angle_3pts(L_SH, L_HIP, L_KNEE)
        right_hip = angle_3pts(R_SH, R_HIP, R_KNEE)
        left_elbow = angle_3pts(L_SH, L_EL, L_WR)
        right_elbow = angle_3pts(R_SH, R_EL, R_WR)
        left_shoulder = angle_3pts(L_EL, L_SH, L_HIP)
        right_shoulder = angle_3pts(R_EL, R_SH, R_HIP)

        left_ankle_vert = L_ANK[1] / float(img_h)
        right_ankle_vert = R_ANK[1] / float(img_h)
        wrist_left = L_WR[1] / float(img_h)
        wrist_right = R_WR[1] / float(img_h)

        # torso bend: angle between torso vector and vertical (upwards)
        torso_vec = np.array([mid_sh[0]-mid_hip[0], mid_sh[1]-mid_hip[1]], dtype=np.float32)
        vertical = np.array([0.0, -1.0], dtype=np.float32)
        denom = (np.linalg.norm(torso_vec) * np.linalg.norm(vertical) + 1e-8)
        cos_t = np.clip(np.dot(torso_vec, vertical) / denom, -1.0, 1.0)
        torso_bend = float(np.degrees(math.acos(cos_t)))

        shoulder_width = dist(L_SH, R_SH) / float(img_w)
        hip_width = dist(L_HIP, R_HIP) / float(img_w)
        torso_length = dist(mid_sh, mid_hip) / float(img_h)
        mid_hip_y = mid_hip[1] / float(img_h)

        # build base row of 17 as in your earlier list
        row = [
            left_knee, right_knee,
            left_hip, right_hip,
            left_elbow, right_elbow,
            left_shoulder, right_shoulder,
            left_ankle_vert, right_ankle_vert,
            wrist_left, wrist_right,
            torso_bend,
            shoulder_width, hip_width, torso_length,
            mid_hip_y
        ]
        rows.append(row)

    data = np.array(rows, dtype=np.float32)  # (N,17)

    # Add vertical displacement features:
    # baseline mid_hip_y at frame0
    baseline_mid_hip_y = data[0, 16] if N>0 else 0.0
    torso_len0 = (data[0, 15] if N>0 else 1.0) + 1e-8
    vert_disp_px = baseline_mid_hip_y - data[:, 16]  # positive = upwards
    vert_disp_norm = vert_disp_px / torso_len0

    data = np.concatenate([data, vert_disp_px.reshape(-1,1), vert_disp_norm.reshape(-1,1)], axis=1)  # now (N,19)

    # Angular velocities for first 13 angle-like features (indices 0..12 of original 17)
    ANGLE_COUNT = 13  # left_knee ... right_shoulder (13)
    ang_vels = []
    # compute difference * fps to approximate angular velocity (deg/s)
    for idx in range(ANGLE_COUNT):
        vals = data[:, idx]
        vel = np.concatenate([[0.0], np.diff(vals)]) * float(fps)
        ang_vels.append(vel.reshape(-1,1))
    ang_vels = np.concatenate(ang_vels, axis=1)  # (N,13)

    data = np.concatenate([data, ang_vels], axis=1)  # (N, 19+13 = 32)
    if data.shape[1] != 32:
        raise RuntimeError(f"Feature size mismatch: expected 32 got {data.shape[1]}")

    return data.astype(np.float32)

# ---------------- windows helper ----------------
def make_windows(features, window=WINDOW_SIZE, step=WINDOW_STEP):
    N, F = features.shape
    windows = []
    centers = []
    start = 0
    while start + window <= N:
        end = start + window
        windows.append(features[start:end])
        centers.append((start + end) // 2)
        start += step
    if not windows:
        return None, None
    return np.stack(windows, axis=0), centers

# ---------------- prediction helpers ----------------
def compute_window_probs(X_windows_norm: np.ndarray, model):
    """
    X_windows_norm: (N_win, T, F) float32 normalized
    model: LSTMClassifier loaded
    """
    ds = WindowDataset(X_windows_norm)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch)    # (B,)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
    if all_probs:
        return np.concatenate(all_probs)
    return np.array([], dtype=np.float32)

def aggregate_frame_probs(n_frames, centers, window_probs, smooth=True):
    accum = np.zeros(n_frames, dtype=np.float32)
    counts = np.zeros(n_frames, dtype=np.float32)
    for c,p in zip(centers, window_probs):
        if 0 <= c < n_frames:
            accum[c] += p
            counts[c] += 1.0
    frame_probs = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame_probs[i] = accum[i] / counts[i] if counts[i] > 0 else 0.0
    if smooth and gaussian_filter1d is not None:
        frame_probs = gaussian_filter1d(frame_probs, sigma=2)
    return frame_probs

# ---------------- annotate/draw ----------------
POSE_CONNECTIONS = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value),
    (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value),
    (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value),
    (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
    (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
    (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value),
    (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
    (mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value),
]

def draw_skeleton_on_frame(frame, kps_frame):
    # kps_frame: (33,3) normalized coords
    h,w = frame.shape[:2]
    # connections
    for a,b in POSE_CONNECTIONS:
        xa, ya = int(kps_frame[a,0]*w), int(kps_frame[a,1]*h)
        xb, yb = int(kps_frame[b,0]*w), int(kps_frame[b,1]*h)
        # basic validity check
        if 0 <= xa < w and 0 <= xb < w and 0 <= ya < h and 0 <= yb < h:
            cv2.line(frame, (xa,ya), (xb,yb), (0,255,0), 2)
    # joints
    for i in range(kps_frame.shape[0]):
        x = int(kps_frame[i,0] * w)
        y = int(kps_frame[i,1] * h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x,y), 3, (0,0,255), -1)

def write_annotated_video(frames, keypoints_seq, frame_probs, out_path, fps):
    if len(frames) == 0:
        raise RuntimeError("No frames to write annotated video.")
    h,w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w,h))
    N = len(frames)
    for i in range(N):
        fr = frames[i].copy()
        # draw skeleton if keypoints available
        if keypoints_seq is not None and i < keypoints_seq.shape[0]:
            draw_skeleton_on_frame(fr, keypoints_seq[i])
        # overlay risk text box
        p = float(frame_probs[i]) if i < len(frame_probs) else 0.0
        label = "SAFE"
        color = (0,255,0)
        if p > 0.7:
            label = "UNSAFE"; color = (0,0,255)
        elif p > 0.4:
            label = "WARNING"; color = (0,255,255)
        cv2.rectangle(fr, (8,8), (380,110), (0,0,0), -1)
        cv2.putText(fr, f"Risk: {p*100:.1f}%", (24,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(fr, label, (24,92), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        writer.write(fr)
    writer.release()

# ---------------- save reports ----------------
def save_window_report(out_prefix, window_probs, summary):
    safe_makedirs(Path(out_prefix).parent)
    csv_win = out_prefix + "_window_probs.csv"
    with open(csv_win, "w") as f:
        f.write("window_index,unsafe_prob\n")
        for i,p in enumerate(window_probs):
            f.write(f"{i},{p:.6f}\n")
    # window plot
    plt.figure(figsize=(8,3))
    plt.plot(window_probs, marker="o")
    plt.ylim(0,1)
    plt.xlabel("Window index"); plt.ylabel("Unsafe prob")
    plt.grid(alpha=0.3)
    plot_win = out_prefix + "_risk_plot.png"
    plt.tight_layout(); plt.savefig(plot_win, dpi=150); plt.close()
    # summary text
    txt = out_prefix + "_summary.txt"
    with open(txt, "w") as f:
        f.write("=== VIDEO-LEVEL RISK SUMMARY ===\n\n")
        f.write(f"Risk level: {summary['risk_level']}\n")
        f.write(f"Mean unsafe probability: {summary['mean_prob']:.6f}\n")
        f.write(f"Max unsafe probability:  {summary['max_prob']:.6f}\n")
        f.write(f"% windows with prob > 0.5: {summary['pct_windows_above_0_5']:.2f}%\n")
    return csv_win, plot_win, txt

def save_frame_csv_and_plot(out_prefix, frame_probs, fps):
    csv_frame = out_prefix + "_frame_probs.csv"
    with open(csv_frame, "w") as f:
        f.write("frame_index,time_sec,unsafe_prob\n")
        for i,p in enumerate(frame_probs):
            f.write(f"{i},{i/float(fps):.4f},{p:.6f}\n")
    # plot
    times = np.arange(len(frame_probs))/float(fps)
    plt.figure(figsize=(10,3))
    plt.plot(times, frame_probs)
    plt.ylim(0,1)
    plt.xlabel("Time (s)"); plt.ylabel("Unsafe prob")
    plt.grid(alpha=0.3)
    plot_frame = out_prefix + "_frame_risk_plot.png"
    plt.tight_layout(); plt.savefig(plot_frame, dpi=150); plt.close()
    return csv_frame, plot_frame

# ---------------- capture webcam helper ----------------
def capture_webcam(duration_sec, out_path, fps=30, camera_index=1):
    out_path = str(out_path)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam for capture.")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w,h))
    start = time.time()
    print(f"Recording {duration_sec}s from webcam to {out_path} ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        if time.time() - start >= duration_sec:
            break
    cap.release(); writer.release()
    print("Saved capture to:", out_path)
    return out_path

# ---------------- main pipelines ----------------
def run_video_pipeline(video_path: str, action: str, model_path: str, out_dir: str):
    video_path = str(video_path)
    out_dir = Path(out_dir)
    safe_makedirs(out_dir)
    video_basename = Path(video_path).stem
    out_prefix = str(out_dir / f"{action}_{video_basename}")

    # check scaler
    scaler_path = find_scaler_for_action(action)
    if scaler_path is None:
        print(f"No scaler found for action '{action}'. Please provide windows/{action}_scaler.pkl or models/{action}_scaler.pkl")
        return

    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return

    print("Extracting keypoints and frames...")
    kps_seq, frames, (h,w), fps = extract_keypoints_from_video(video_path)
    n_frames = kps_seq.shape[0]
    if n_frames == 0:
        print("No frames found in video.")
        return

    print(f"Frames: {n_frames}  FPS: {fps}  Resolution: {w}x{h}")

    print("Extracting features per frame...")
    feats = extract_features_from_keypoints_seq(kps_seq, h, w, fps)
    print("Features shape:", feats.shape)

    print("Creating sliding windows...")
    X_win, centers = make_windows(feats, WINDOW_SIZE, WINDOW_STEP)
    if X_win is None:
        print("Not enough frames for windows (need at least WINDOW_SIZE frames).")
        return
    print("Windows:", X_win.shape, "Centers:", len(centers))

    # load scaler and normalize
    print("Loading scaler:", scaler_path)
    scaler = joblib.load(scaler_path)
    N,T,F = X_win.shape
    flat = X_win.reshape(-1, F)
    flat_norm = scaler.transform(flat)  # may produce float64
    X_norm = flat_norm.reshape(N, T, F).astype(np.float32)

    # load model
    print("Loading model:", model_path)
    state = torch.load(str(model_path), map_location=DEVICE)
    model = LSTMClassifier(feat_dim=F).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    # compute window probs
    print("Computing window probabilities...")
    window_probs = compute_window_probs(X_norm, model)
    print("Window probs stats: mean", float(window_probs.mean()), "max", float(window_probs.max()))

    # compute frame probs aggregated
    frame_probs = aggregate_frame_probs(n_frames, centers, window_probs, smooth=True)
    # write annotated video with skeleton + frame overlays
    annotated_video = out_prefix + "_annotated.mp4"
    print("Writing annotated video:", annotated_video)
    # For drawing, the keypoints must be normalized in [0,1] format - they already are from extraction
    write_annotated_video(frames, kps_seq, frame_probs, annotated_video, fps)

    # save CSVs and plots
    summary = {
        "mean_prob": float(np.mean(window_probs)) if window_probs.size else 0.0,
        "max_prob": float(np.max(window_probs)) if window_probs.size else 0.0,
        "pct_windows_above_0_5": float((window_probs > 0.5).mean() * 100.0) if window_probs.size else 0.0,
        "risk_level": ("SAFE" if float(np.mean(window_probs))<0.2 else ("MODERATE RISK" if float(np.mean(window_probs))<0.5 else "HIGH RISK"))
    }
    csv_win, plot_win, txt_summary = save_window_report(out_prefix, window_probs, summary)
    csv_frame, plot_frame = save_frame_csv_and_plot(out_prefix, frame_probs, fps)

    print("Outputs saved with prefix:", out_prefix)
    print("Annotated video:", annotated_video)
    print("Window CSV:", csv_win, "Frame CSV:", csv_frame)
    print("Plots:", plot_win, plot_frame)
    print("Summary:", txt_summary)

def run_npz_pipeline(windows_npz: str, model_path: str, out_dir: str):
    windows_npz = str(windows_npz)
    out_dir = Path(out_dir)
    safe_makedirs(out_dir)
    base = Path(windows_npz).stem
    out_prefix = str(out_dir / base)
    if not Path(windows_npz).exists():
        print("NPZ windows file not found:", windows_npz)
        return
    d = np.load(windows_npz)
    if "data" not in d:
        print("NPZ must contain 'data' key with shape (N,T,F)")
        return
    X = d["data"].astype(np.float32)
    N, T, F = X.shape
    print("Loaded windows:", X.shape, "min", float(X.min()), "max", float(X.max()))
    if not Path(model_path).exists():
        print("Model not found:", model_path); return
    state = torch.load(str(model_path), map_location=DEVICE)
    model = LSTMClassifier(feat_dim=F).to(DEVICE)
    model.load_state_dict(state)
    window_probs = compute_window_probs(X, model)
    summary = {
        "mean_prob": float(window_probs.mean()) if window_probs.size else 0.0,
        "max_prob": float(window_probs.max()) if window_probs.size else 0.0,
        "pct_windows_above_0_5": float((window_probs > 0.5).mean() * 100.0) if window_probs.size else 0.0,
        "risk_level": ("SAFE" if float(window_probs.mean())<0.2 else ("MODERATE RISK" if float(window_probs.mean())<0.5 else "HIGH RISK"))
    }
    save_window_report(out_prefix, window_probs, summary)
    print("Saved NPZ report with prefix:", out_prefix)

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, help="Path to input .mp4 video")
    p.add_argument("--action", type=str, choices=["attack","block","defence"], help="Action (required with --video or --capture)")
    p.add_argument("--windows_npz", type=str, help="Path to windows NPZ (pre-normalized)")
    p.add_argument("--capture", type=int, help="Record N seconds from webcam and process (requires --action)")
    p.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to classifier model (.pth)")
    p.add_argument("--out_dir", type=str, default=str(BASE_DIR / "reports"), help="Directory to write reports")
    return p.parse_args()

def main():
    args = parse_args()

    # ensure out_dir exists
    safe_makedirs(Path(args.out_dir))

    # If capture requested, record and then set args.video
    if args.capture:
        if not args.action:
            print("--action is required when using --capture")
            return
        ts = int(time.time())
        out_name = f"capture_{args.action}_{ts}.mp4"
        out_path = Path(args.out_dir) / out_name
        try:
            capture_webcam(int(args.capture), out_path)
        except Exception as e:
            print("Webcam capture failed:", str(e))
            return
        args.video = str(out_path)

    # Video mode
    if args.video:
        if not args.action:
            print("--action is required when processing a video.")
            return
        run_video_pipeline(args.video, args.action, args.model, args.out_dir)
        return

    # NPZ mode
    if args.windows_npz:
        run_npz_pipeline(args.windows_npz, args.model, args.out_dir)
        return

    print("Usage: provide --video and --action, or --capture N and --action, or --windows_npz")

if __name__ == "__main__":
    main()
