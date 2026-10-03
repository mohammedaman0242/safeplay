# extract_keypoints_expanded.py
import os, cv2, numpy as np, mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIDE_VIEW_DIR = os.path.join(BASE_DIR, "sample_videos")
OUT_BASE = os.path.join(BASE_DIR, "sample_videos/key_p")
os.makedirs(OUT_BASE, exist_ok=True)

mp_pose = mp.solutions.pose

# Expanded landmarks (includes heel and foot index)
LANDMARKS = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
]
LM_IDX = [lm.value for lm in LANDMARKS]
LM_NAMES = [lm.name.lower() for lm in LANDMARKS]  # for metadata if needed

def extract_and_save(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            arr = np.full((len(LM_IDX), 4), np.nan, dtype=np.float32)  # x,y,z,vis
            if res.pose_landmarks:
                for i, idx in enumerate(LM_IDX):
                    lm = res.pose_landmarks.landmark[idx]
                    arr[i, 0] = lm.x * w
                    arr[i, 1] = lm.y * h
                    arr[i, 2] = lm.z * max(h,w)   # z scaled to pixel-ish units (optional)
                    arr[i, 3] = lm.visibility
            frames.append(arr)
    cap.release()
    data = np.stack(frames, axis=0)  # (F, L, 4)
    np.savez_compressed(out_path, keypoints=data, fps=float(fps), video=os.path.basename(video_path))
    print(f"Saved {out_path}  shape={data.shape}, fps={fps}")

def main():
    ACTIONS = ["attack","block","defence"]
    for action in ACTIONS:
        in_folder = os.path.join(SIDE_VIEW_DIR, action)
        out_folder = os.path.join(OUT_BASE, action)
        os.makedirs(out_folder, exist_ok=True)
        if not os.path.isdir(in_folder):
            print("Skipping missing:", in_folder)
            continue
        print("Processing:", action)
        for fname in sorted(os.listdir(in_folder)):
            if not fname.lower().endswith(".mp4"):
                continue
            in_path = os.path.join(in_folder, fname)
            out_name = fname.replace(".mp4", "_kps.npz")
            out_path = os.path.join(out_folder, out_name)
            extract_and_save(in_path, out_path)
    print("All done.")

if __name__ == '__main__':
    main()
