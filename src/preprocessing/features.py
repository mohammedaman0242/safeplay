"""
Canonical SafePlay feature extraction.

Input:
    keypoints: (frames, 17, 4)
    channels: [x_px, y_px, z, visibility]

The 32 features exactly follow the original SafePlay training pipeline:

    17 base features
    + 2 vertical displacement features
    + 13 angular velocity features
    = 32 features
"""

import numpy as np
from math import atan2, degrees, hypot


EXPECTED_FEATURE_DIM = 32


FEATURE_NAMES = [
    # 17 base features
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "left_elbow",
    "right_elbow",
    "left_shoulder",
    "right_shoulder",
    "left_ankle_vert",
    "right_ankle_vert",
    "wrist_left",
    "wrist_right",
    "torso_bend",
    "shoulder_width",
    "hip_width",
    "torso_length",
    "mid_hip_y",

    # 2 displacement features
    "vert_disp_px",
    "vert_disp_norm",

    # 13 angular velocity features
    "left_knee_vel",
    "right_knee_vel",
    "left_hip_vel",
    "right_hip_vel",
    "left_elbow_vel",
    "right_elbow_vel",
    "left_shoulder_vel",
    "right_shoulder_vel",
    "left_ankle_vert_vel",
    "right_ankle_vert_vel",
    "wrist_left_vel",
    "wrist_right_vel",
    "torso_bend_vel",
]


LANDMARK_NAMES = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


ANGLE_FEATURE_NAMES = [
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "left_elbow",
    "right_elbow",
    "left_shoulder",
    "right_shoulder",
    "left_ankle_vert",
    "right_ankle_vert",
    "wrist_left",
    "wrist_right",
    "torso_bend",
]


def get_feature_names():
    return FEATURE_NAMES.copy()


def validate_features(data):
    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrix, got shape {data.shape}"
        )

    if data.shape[1] != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_DIM} features, "
            f"got {data.shape[1]}"
        )

    return True


def safe_point(frame, index):
    """Return (x, y) or None if invalid."""
    x = frame[index, 0]
    y = frame[index, 1]

    if np.isnan(x) or np.isnan(y):
        return None

    return float(x), float(y)


def midpoint(a, b):
    if a is None or b is None:
        return np.nan, np.nan

    return (
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
    )


def distance(a, b):
    if a is None or b is None:
        return np.nan

    return hypot(
        a[0] - b[0],
        a[1] - b[1],
    )


def angle_3_points(a, b, c):
    """
    Angle at point b formed by a-b-c.
    """
    if a is None or b is None or c is None:
        return np.nan

    v1 = (
        a[0] - b[0],
        a[1] - b[1],
    )

    v2 = (
        c[0] - b[0],
        c[1] - b[1],
    )

    n1 = hypot(v1[0], v1[1])
    n2 = hypot(v2[0], v2[1])

    if n1 * n2 < 1e-8:
        return np.nan

    cosine = (
        (v1[0] * v2[0] + v1[1] * v2[1])
        / (n1 * n2)
    )

    cosine = max(-1.0, min(1.0, cosine))

    return float(degrees(np.arccos(cosine)))


def angle_from_vertical(a, b):
    """
    Angle of vector a -> b relative to vertical.

    0 degrees = pointing upward.
    """
    if a is None or b is None:
        return np.nan

    vx = b[0] - a[0]
    vy = b[1] - a[1]

    return float(degrees(atan2(vx, -vy)))


def extract_features(kps4, fps=30.0):
    """
    Convert smoothed SafePlay keypoints into 32 features.

    Parameters
    ----------
    kps4 : np.ndarray
        Shape: (frames, 17, 4)
        Channels: x, y, z, visibility

    fps : float
        Video frame rate.

    Returns
    -------
    data : np.ndarray
        Shape: (frames, 32)

    feature_names : list[str]
        Names of all 32 features.
    """

    kps4 = np.asarray(kps4, dtype=np.float32)

    if kps4.ndim != 3:
        raise ValueError(
            f"Expected keypoints with 3 dimensions, got {kps4.shape}"
        )

    if kps4.shape[1] != 17:
        raise ValueError(
            f"Expected 17 landmarks, got {kps4.shape[1]}"
        )

    frames = kps4.shape[0]

    rows = []

    for frame_idx in range(frames):

        frame = kps4[frame_idx]

        def P(name):
            return safe_point(
                frame,
                LANDMARK_NAMES.index(name)
            )

        left_shoulder = P("left_shoulder")
        right_shoulder = P("right_shoulder")

        left_elbow = P("left_elbow")
        right_elbow = P("right_elbow")

        left_wrist = P("left_wrist")
        right_wrist = P("right_wrist")

        left_hip = P("left_hip")
        right_hip = P("right_hip")

        left_knee = P("left_knee")
        right_knee = P("right_knee")

        left_ankle = P("left_ankle")
        right_ankle = P("right_ankle")

        mid_shoulder = midpoint(
            left_shoulder,
            right_shoulder
        )

        mid_hip = midpoint(
            left_hip,
            right_hip
        )

        # Joint angles
        left_knee_angle = angle_3_points(
            left_hip,
            left_knee,
            left_ankle
        )

        right_knee_angle = angle_3_points(
            right_hip,
            right_knee,
            right_ankle
        )

        left_hip_angle = angle_3_points(
            left_shoulder,
            left_hip,
            left_knee
        )

        right_hip_angle = angle_3_points(
            right_shoulder,
            right_hip,
            right_knee
        )

        left_elbow_angle = angle_3_points(
            left_shoulder,
            left_elbow,
            left_wrist
        )

        right_elbow_angle = angle_3_points(
            right_shoulder,
            right_elbow,
            right_wrist
        )

        left_shoulder_angle = angle_3_points(
            left_elbow,
            left_shoulder,
            left_hip
        )

        right_shoulder_angle = angle_3_points(
            right_elbow,
            right_shoulder,
            right_hip
        )

        # Orientation features
        left_ankle_vert = angle_from_vertical(
            left_knee,
            left_ankle
        )

        right_ankle_vert = angle_from_vertical(
            right_knee,
            right_ankle
        )

        wrist_left = angle_from_vertical(
            left_elbow,
            left_wrist
        )

        wrist_right = angle_from_vertical(
            right_elbow,
            right_wrist
        )

        torso_bend = angle_from_vertical(
            mid_hip,
            mid_shoulder
        )

        # Body dimensions in pixel coordinates
        shoulder_width = distance(
            left_shoulder,
            right_shoulder
        )

        hip_width = distance(
            left_hip,
            right_hip
        )

        torso_length = distance(
            mid_shoulder,
            mid_hip
        )

        mid_hip_y = (
            mid_hip[1]
            if not np.isnan(mid_hip[1])
            else np.nan
        )

        rows.append([
            left_knee_angle,
            right_knee_angle,
            left_hip_angle,
            right_hip_angle,
            left_elbow_angle,
            right_elbow_angle,
            left_shoulder_angle,
            right_shoulder_angle,
            left_ankle_vert,
            right_ankle_vert,
            wrist_left,
            wrist_right,
            torso_bend,
            shoulder_width,
            hip_width,
            torso_length,
            mid_hip_y,
        ])

    data = np.asarray(rows, dtype=np.float32)

    if frames == 0:
        return (
            np.empty((0, EXPECTED_FEATURE_DIM), dtype=np.float32),
            FEATURE_NAMES.copy(),
        )

    # ---------------------------------------------------------
    # Vertical displacement
    # ---------------------------------------------------------

    mid_hip_idx = FEATURE_NAMES.index("mid_hip_y")

    baseline_n = max(1, int(0.1 * frames))

    baseline_values = data[:baseline_n, 16]

    finite_baseline = baseline_values[
        np.isfinite(baseline_values)
        ]

    if len(finite_baseline) > 0:
        baseline = np.median(finite_baseline)
    else:
    # Fall back to the first finite mid-hip value
    # anywhere in the sequence.
        all_finite = data[:, 16][
            np.isfinite(data[:, 16])
        ]

        if len(all_finite) > 0:
            baseline = np.median(all_finite)
        else:
            baseline = np.nan

    vert_disp_px = (
        baseline - data[:, 16]
    )

    torso_length_idx = 15

    torso_length = data[:, torso_length_idx]

    vert_disp_norm = (
        vert_disp_px /
        (torso_length + 1e-8)
    )

    data = np.concatenate(
        [
            data,
            vert_disp_px.reshape(-1, 1),
            vert_disp_norm.reshape(-1, 1),
        ],
        axis=1,
    )

    # ---------------------------------------------------------
    # Angular velocities
    # ---------------------------------------------------------

    for feature_name in ANGLE_FEATURE_NAMES:

        idx = FEATURE_NAMES.index(feature_name)

        values = data[:, idx]

        velocity = (
            np.concatenate(
                [
                    [0.0],
                    np.diff(values),
                ]
            )
            * float(fps)
        )

        data = np.concatenate(
            [
                data,
                velocity.reshape(-1, 1),
            ],
            axis=1,
        )

    validate_features(data)

    return data.astype(np.float32), FEATURE_NAMES.copy()