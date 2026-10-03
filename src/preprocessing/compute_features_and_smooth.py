import os
import numpy as np
from scipy.signal import savgol_filter

from src.preprocessing.features import extract_features


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

KEYPOINT_DIR = os.path.join(
    BASE_DIR, "data", "processed", "keypoints"
)

FEATURE_DIR = os.path.join(
    BASE_DIR, "data", "processed", "features"
)

LABELS = ["safe", "unsafe"]
ACTIONS = ["attack", "block", "defence"]

# Maximum number of consecutive missing frames that we will
# interpolate.
MAX_INTERPOLATION_GAP = 10


# ============================================================
# Utility functions
# ============================================================

def get_missing_frame_mask(keypoints):
    """
    A frame is considered missing if any x/y landmark coordinate
    contains NaN.
    """
    xy = keypoints[:, :, :2]
    return np.isnan(xy).any(axis=(1, 2))


def find_missing_runs(mask):
    """
    Return consecutive True runs as:
        [(start, end, length), ...]
    """
    runs = []

    start = None

    for i, value in enumerate(mask):
        if value and start is None:
            start = i

        elif not value and start is not None:
            runs.append((start, i - 1, i - start))
            start = None

    if start is not None:
        runs.append(
            (start, len(mask) - 1, len(mask) - start)
        )

    return runs


def interpolate_short_gaps(keypoints, max_gap=10):
    """
    Interpolate x/y coordinates for short missing-pose gaps.

    Long gaps are intentionally left as NaN.

    Returns:
        repaired_keypoints
        repaired_mask
        remaining_invalid_mask
    """

    repaired = keypoints.copy()

    original_missing = get_missing_frame_mask(repaired)

    repaired_mask = np.zeros(
        len(repaired),
        dtype=bool
    )

    for start, end, length in find_missing_runs(original_missing):

        if length > max_gap:
            continue

        # Need valid frames on both sides of the gap.
        left = start - 1
        right = end + 1

        if left < 0 or right >= len(repaired):
            continue

        # If neighboring frames are also invalid, skip.
        if original_missing[left] or original_missing[right]:
            continue

        # Interpolate x/y for every landmark.
        for frame in range(start, end + 1):

            alpha = (frame - left) / (right - left)

            repaired[frame, :, 0] = (
                repaired[left, :, 0]
                + alpha
                * (
                    repaired[right, :, 0]
                    - repaired[left, :, 0]
                )
            )

            repaired[frame, :, 1] = (
                repaired[left, :, 1]
                + alpha
                * (
                    repaired[right, :, 1]
                    - repaired[left, :, 1]
                )
            )

            repaired_mask[frame] = True

    remaining_invalid = get_missing_frame_mask(repaired)

    return repaired, repaired_mask, remaining_invalid


def smooth_features(features, valid_mask):
    """
    Apply Savitzky-Golay smoothing only to contiguous valid
    feature regions.

    Invalid frames remain invalid.
    """

    result = features.copy()

    window_length = 7
    polyorder = 2

    for col in range(result.shape[1]):

        values = result[:, col].copy()

        valid = (
            valid_mask
            & np.isfinite(values)
        )

        start = None

        for i in range(len(values) + 1):

            is_valid = (
                i < len(values)
                and valid[i]
            )

            if is_valid and start is None:
                start = i

            elif not is_valid and start is not None:

                end = i
                segment = values[start:end]

                if len(segment) >= window_length:

                    w = min(
                        window_length,
                        len(segment)
                    )

                    if w % 2 == 0:
                        w -= 1

                    if w > polyorder:

                        result[start:end, col] = (
                            savgol_filter(
                                segment,
                                window_length=w,
                                polyorder=polyorder
                            )
                        )

                start = None

    return result


# ============================================================
# Main processing
# ============================================================

def main():

    processed = 0
    failed = 0

    total_original_missing = 0
    total_repaired = 0
    total_remaining_invalid = 0

    os.makedirs(FEATURE_DIR, exist_ok=True)

    print("=" * 70)
    print("SAFEPLAY FEATURE EXTRACTION")
    print("=" * 70)
    print(
        f"Maximum interpolation gap: "
        f"{MAX_INTERPOLATION_GAP} frames"
    )
    print()

    for label in LABELS:

        for action in ACTIONS:

            input_dir = os.path.join(
                KEYPOINT_DIR,
                label,
                action
            )

            output_dir = os.path.join(
                FEATURE_DIR,
                label,
                action
            )

            os.makedirs(output_dir, exist_ok=True)

            if not os.path.isdir(input_dir):
                continue

            for filename in sorted(os.listdir(input_dir)):

                if not filename.endswith(".npz"):
                    continue

                input_path = os.path.join(
                    input_dir,
                    filename
                )

                output_name = (
                    filename.replace(
                        "_kps.npz",
                        "_features.npz"
                    )
                )

                output_path = os.path.join(
                    output_dir,
                    output_name
                )

                try:

                    data = np.load(
                        input_path,
                        allow_pickle=True
                    )

                    keypoints = data["keypoints"].astype(
                        np.float32
                    )

                    fps = float(data["fps"])

                    original_missing = (
                        get_missing_frame_mask(keypoints)
                    )

                    original_missing_count = int(
                        original_missing.sum()
                    )

                    # ------------------------------------------------
                    # Repair short missing-pose gaps
                    # ------------------------------------------------

                    repaired_keypoints, repaired_mask, remaining_invalid = (
                        interpolate_short_gaps(
                            keypoints,
                            MAX_INTERPOLATION_GAP
                        )
                    )

                    repaired_count = int(
                        repaired_mask.sum()
                    )

                    remaining_invalid_count = int(
                        remaining_invalid.sum()
                    )

                    # ------------------------------------------------
                    # Feature extraction
                    # ------------------------------------------------

                    features, feature_names = extract_features(
                        repaired_keypoints,
                        fps
                    )

                    # ------------------------------------------------
                    # Smooth valid contiguous regions
                    # ------------------------------------------------

                    valid_mask = ~remaining_invalid
                    
                    features = smooth_features(
                            features,
                            valid_mask
                    )

                    # ------------------------------------------------
                    # Save
                    # ------------------------------------------------

                    np.savez_compressed(
                        output_path,
                        data=features.astype(np.float32),
                        fps=np.float32(fps),
                        video=filename,
                        label=label,
                        action=action,
                        valid_mask=(
                            ~remaining_invalid
                        ),
                        repaired_mask=repaired_mask,
                        original_missing_mask=original_missing
                    )

                    processed += 1

                    total_original_missing += (
                        original_missing_count
                    )

                    total_repaired += repaired_count

                    total_remaining_invalid += (
                        remaining_invalid_count
                    )

                    print(
                        f"[{label.upper():6} | {action:7}] "
                        f"{filename:18} "
                        f"frames={len(keypoints):4} "
                        f"missing={original_missing_count:4} "
                        f"repaired={repaired_count:4} "
                        f"remaining={remaining_invalid_count:4}"
                    )

                except Exception as exc:

                    failed += 1

                    print(
                        f"[ERROR] "
                        f"{label}/{action}/{filename}: "
                        f"{exc}"
                    )

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"Processed videos:          {processed}")
    print(f"Failed videos:             {failed}")
    print(f"Original missing frames:   {total_original_missing}")
    print(f"Frames interpolated:       {total_repaired}")
    print(f"Invalid frames remaining:  {total_remaining_invalid}")
    print("=" * 70)


if __name__ == "__main__":
    main()