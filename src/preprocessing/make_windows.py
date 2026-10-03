import os
import numpy as np


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

FEATURE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "features"
)

WINDOW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "windows"
)

LABELS = ["safe", "unsafe"]
ACTIONS = ["attack", "block", "defence"]

SEQ_LEN = 30
STEP = 15
EXPECTED_FEATURES = 32


# ============================================================
# Window generation
# ============================================================

def create_windows(features, valid_mask):
    """
    Create fixed-length temporal windows.

    A window is accepted only when:
        1. It contains exactly SEQ_LEN frames.
        2. Every frame is marked valid.
        3. All feature values are finite.
    """

    windows = []
    starts = []

    num_frames = len(features)

    for start in range(
        0,
        num_frames - SEQ_LEN + 1,
        STEP
    ):

        end = start + SEQ_LEN

        window = features[start:end]
        mask = valid_mask[start:end]

        if not mask.all():
            continue

        if not np.isfinite(window).all():
            continue

        windows.append(window.astype(np.float32))
        starts.append(start)

    if not windows:
        return (
            np.empty(
                (0, SEQ_LEN, EXPECTED_FEATURES),
                dtype=np.float32
            ),
            np.empty((0,), dtype=np.int32)
        )

    return (
        np.stack(windows),
        np.asarray(starts, dtype=np.int32)
    )


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        WINDOW_DIR,
        exist_ok=True
    )

    total_files = 0
    total_windows = 0
    total_discarded = 0

    print("=" * 70)
    print("SAFEPLAY WINDOW GENERATION")
    print("=" * 70)

    for label in LABELS:

        for action in ACTIONS:

            input_dir = os.path.join(
                FEATURE_DIR,
                label,
                action
            )

            output_dir = os.path.join(
                WINDOW_DIR,
                label,
                action
            )

            os.makedirs(
                output_dir,
                exist_ok=True
            )

            if not os.path.isdir(input_dir):
                continue

            for filename in sorted(
                os.listdir(input_dir)
            ):

                if not filename.endswith(
                    "_features.npz"
                ):
                    continue

                input_path = os.path.join(
                    input_dir,
                    filename
                )

                data = np.load(
                    input_path,
                    allow_pickle=True
                )

                features = data["data"]
                valid_mask = data["valid_mask"]

                if features.ndim != 2:
                    print(
                        f"[SKIP] {filename}: "
                        f"unexpected shape {features.shape}"
                    )
                    continue

                if features.shape[1] != EXPECTED_FEATURES:
                    print(
                        f"[SKIP] {filename}: "
                        f"expected {EXPECTED_FEATURES} "
                        f"features, got "
                        f"{features.shape[1]}"
                    )
                    continue

                windows, starts = create_windows(
                    features,
                    valid_mask
                )

                possible_windows = max(
                    0,
                    (
                        len(features) - SEQ_LEN
                    ) // STEP + 1
                )

                discarded = (
                    possible_windows
                    - len(windows)
                )

                video_id = filename.replace(
                    "_features.npz",
                    ""
                )

                output_name = (
                    f"{video_id}_windows.npz"
                )

                output_path = os.path.join(
                    output_dir,
                    output_name
                )

                np.savez_compressed(
                    output_path,
                    windows=windows,
                    starts=starts,
                    label=label,
                    action=action,
                    video_id=video_id,
                    seq_len=SEQ_LEN,
                    step=STEP
                )

                total_files += 1
                total_windows += len(windows)
                total_discarded += discarded

                print(
                    f"[{label.upper():6} | {action:7}] "
                    f"{video_id:5} "
                    f"frames={len(features):4} "
                    f"windows={len(windows):3} "
                    f"discarded={discarded:3}"
                )

    print()
    print("=" * 70)
    print("WINDOW GENERATION COMPLETE")
    print("=" * 70)
    print(f"Feature files processed: {total_files}")
    print(f"Valid windows created:    {total_windows}")
    print(f"Windows discarded:        {total_discarded}")
    print("=" * 70)


if __name__ == "__main__":
    main()