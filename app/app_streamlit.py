import streamlit as st
import os
import sys
import tempfile
import subprocess
from pathlib import Path
import time


# Project root: safeplay/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main inference pipeline
FULL_SCRIPT = PROJECT_ROOT / "src" / "inference" / "full_report_from_video.py"

st.set_page_config(
    page_title="SafePlay — Volleyball Motion Risk Analysis",
    page_icon="🏐",
    layout="wide",
)

st.title("🏐 SafePlay — Volleyball Motion Risk Analysis")
st.caption(
    "Computer vision system for analyzing volleyball movements "
    "and identifying potentially risky motion patterns."
)

# Choose input mode
mode = st.radio(
    "Choose input",
    ["Upload video", "Realtime capture (record N seconds)"]
)

# Action selection
action = st.selectbox(
    "Action type",
    ["attack", "block", "defence"]
)

# Output folder
default_output = PROJECT_ROOT / "results" / "streamlit_runs"

out_folder = st.text_input(
    "Output folder",
    str(default_output)
)

out_folder = Path(out_folder).resolve()
out_folder.mkdir(parents=True, exist_ok=True)

# Use the same Python environment running Streamlit
python_exe = sys.executable

tmpdir = (
    Path(tempfile.gettempdir())
    / f"safeplay_demo_{int(time.time())}"
)
tmpdir.mkdir(parents=True, exist_ok=True)

video_path = None


# ---------------------------------------------------------
# Upload video
# ---------------------------------------------------------

if mode == "Upload video":

    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi"],
        accept_multiple_files=False,
    )

    if uploaded is not None:

        save_to = tmpdir / uploaded.name

        with open(save_to, "wb") as f:
            f.write(uploaded.getbuffer())

        video_path = save_to

        st.success(f"Video uploaded: {uploaded.name}")


# ---------------------------------------------------------
# Realtime capture
# ---------------------------------------------------------

elif mode == "Realtime capture (record N seconds)":

    seconds = st.number_input(
        "Record duration (seconds)",
        min_value=1,
        max_value=60,
        value=6,
        step=1,
    )

    if st.button(f"Record {seconds}s from webcam"):

        st.info(
            "Recording from webcam. Please remain visible to the camera."
        )

        cmd = [
            python_exe,
            str(FULL_SCRIPT),
            "--capture",
            str(seconds),
            "--action",
            action,
            "--out_dir",
            str(out_folder),
        ]

        with st.spinner("Recording and processing..."):

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT),
            )

        st.text(proc.stdout)

        if proc.returncode == 0:
            st.success("Recording and processing completed.")
        else:
            st.error(
                f"Processing failed with exit code {proc.returncode}."
            )


# ---------------------------------------------------------
# Process uploaded video
# ---------------------------------------------------------

if video_path is not None:

    if st.button("Process uploaded video"):

        st.info("Processing video...")

        cmd = [
            python_exe,
            str(FULL_SCRIPT),
            "--video",
            str(video_path),
            "--action",
            action,
            "--out_dir",
            str(out_folder),
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        log_area = st.empty()
        out_lines = []

        for line in proc.stdout:
            out_lines.append(line)
            log_area.text("".join(out_lines[-200:]))

        proc.wait()

        if proc.returncode == 0:
            st.success("Processing completed.")
        else:
            st.error(
                f"Processing failed with exit code {proc.returncode}."
            )


# ---------------------------------------------------------
# Display generated results
# ---------------------------------------------------------

st.header("Latest Results")

reports = sorted(
    out_folder.glob("*"),
    key=os.path.getmtime,
    reverse=True,
)

if not reports:

    st.info(
        "No results yet. Upload a video or record a capture to begin."
    )

else:

    shown = 0

    for file_path in reports:

        if shown >= 12:
            break

        filename = file_path.name.lower()

        if filename.endswith(
            (".png", ".jpg", ".jpeg", ".mp4", ".csv", ".txt", ".pdf")
        ):

            st.write(file_path.name)

            if filename.endswith(".mp4"):

                st.video(str(file_path))

            elif filename.endswith(
                (".png", ".jpg", ".jpeg")
            ):

                st.image(str(file_path))

            else:

                with open(file_path, "rb") as f:
                    data = f.read()

                st.download_button(
                    "Download",
                    data=data,
                    file_name=file_path.name,
                    key=str(file_path),
                )

            shown += 1