import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    _CLICK = True
except ImportError:
    _CLICK = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blur_selected import (
    detect_faces,
    get_embedding,
    load_detector,
    load_embedder,
    pick_box_for_point,
    track_run,
)

# ── Constants ──────────────────────────────────────────────────────────────────
SUPPORTED = ["mp4", "avi", "mov", "mkv", "webm"]
MAX_MB     = 500
DISPLAY_W  = 800

# ── Page ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Face Blur", layout="wide")
st.title("Face Blur")
st.caption(
    "Upload a video · scrub to a clear frame · click the face · download the result."
)

# ── Models (cached across reruns) ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading detection models…")
def _models():
    return load_detector(), load_embedder()

# ── Helpers ────────────────────────────────────────────────────────────────────
def _load_frames(path):
    cap = cv2.VideoCapture(path)
    meta = {
        "fps":    cap.get(cv2.CAP_PROP_FPS) or 25,
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    frames = []
    bar = st.progress(0, text="Decoding frames…")
    total_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        if len(frames) % 60 == 0:
            bar.progress(min(len(frames) / total_hint, 0.99), text=f"Decoded {len(frames)} frames…")
    cap.release()
    bar.empty()
    return frames, meta

def _to_pil(bgr_frame, max_w=DISPLAY_W):
    """Return (PIL image, scale) where scale = displayed_w / original_w."""
    h, w = bgr_frame.shape[:2]
    scale = min(1.0, max_w / w)
    if scale < 1.0:
        bgr_frame = cv2.resize(bgr_frame, (int(w * scale), int(h * scale)))
    return Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)), scale

# ── Sidebar: limits & formats ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Info")
    st.markdown(f"""
**Supported formats**
{" · ".join(e.upper() for e in SUPPORTED)}

**File size limit**
{MAX_MB} MB

**Memory usage**
The full video is decoded into RAM.
Expect ~1–4 GB for a typical 100 MB clip.

**Face detection model**
OpenCV SSD (res10 300×300)

**Face recognition model**
OpenFace 128-d embeddings
""")
    if not _CLICK:
        st.warning(
            "Click-to-select unavailable.  \n"
            "Install for a better experience:  \n"
            "`pip install streamlit-image-coordinates`"
        )

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    f"Upload video ({', '.join(e.upper() for e in SUPPORTED)} · max {MAX_MB} MB)",
    type=SUPPORTED,
)

if uploaded is None:
    st.stop()

if uploaded.size > MAX_MB * 1024 * 1024:
    st.error(f"File is {uploaded.size / 1024**2:.1f} MB — limit is {MAX_MB} MB.")
    st.stop()

# Save to disk if this is a new upload
if st.session_state.get("upload_name") != uploaded.name:
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.close()
    for k in ("frames", "meta", "sel_frame", "sel_box"):
        st.session_state.pop(k, None)
    st.session_state.upload_name = uploaded.name
    st.session_state.tmp_path    = tmp.name

# Decode frames once
if "frames" not in st.session_state:
    frames, meta = _load_frames(st.session_state.tmp_path)
    if not frames:
        st.error("Could not read any frames — is this a valid video file?")
        st.stop()
    st.session_state.frames = frames
    st.session_state.meta   = meta

frames: list = st.session_state.frames
meta:   dict = st.session_state.meta
total        = len(frames)
fps          = meta["fps"]

# ── Video metadata ─────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Frames",     total)
c2.metric("Resolution", f"{meta['width']}×{meta['height']}")
c3.metric("FPS",        f"{fps:.2f}")
c4.metric("Duration",   f"{total / fps:.1f} s")
c5.metric("File size",  f"{uploaded.size / 1024**2:.1f} MB")

st.divider()

# ── Step 1: scrub ──────────────────────────────────────────────────────────────
st.subheader("Step 1 — Find a frame with a clear, frontal view of the face")
st.caption("Frontal or near-frontal frames give the best identity match.")
frame_idx = st.slider("Frame", 0, total - 1, min(30, total - 1), key="scrub")

detector, embedder = _models()

preview = frames[frame_idx].copy()
boxes   = detect_faces(detector, preview, confidence=0.4)
for i, b in enumerate(boxes):
    cv2.rectangle(preview, (b[0], b[1]), (b[2], b[3]), (0, 210, 0), 2)
    cv2.putText(
        preview, str(i + 1), (b[0] + 4, b[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 0), 2,
    )

pil_preview, scale = _to_pil(preview)

# ── Step 2: select face ────────────────────────────────────────────────────────
st.subheader("Step 2 — Select the face to blur")

if _CLICK:
    st.caption("Click directly on the face in the image below.")
    coords = streamlit_image_coordinates(pil_preview, key=f"sel_{frame_idx}")
    if coords is not None:
        orig_x = int(coords["x"] / scale)
        orig_y = int(coords["y"] / scale)
        box = pick_box_for_point(boxes, (orig_x, orig_y))
        if box:
            st.session_state.sel_frame = frame_idx
            st.session_state.sel_box   = box
        else:
            st.warning("No face found near that click — try clicking closer to a green box.")
else:
    st.image(pil_preview, use_container_width=False)
    if boxes:
        choice = st.radio(
            "Which face?",
            [f"Face {i + 1}" for i in range(len(boxes))],
            horizontal=True,
            key=f"radio_{frame_idx}",
        )
        if st.button("Confirm selection"):
            idx_chosen = int(choice.split()[-1]) - 1
            st.session_state.sel_frame = frame_idx
            st.session_state.sel_box   = boxes[idx_chosen]
    else:
        st.warning(
            "No faces detected in this frame. "
            "Try a different frame or one where the subject is facing the camera."
        )

# Confirmation thumbnail
if st.session_state.get("sel_box") is not None:
    sf    = st.session_state.sel_frame
    sb    = st.session_state.sel_box
    thumb = frames[sf].copy()
    cv2.rectangle(thumb, (sb[0], sb[1]), (sb[2], sb[3]), (0, 60, 220), 3)
    pil_thumb, _ = _to_pil(thumb, max_w=380)
    st.success(f"Face locked at frame {sf}.")
    st.image(pil_thumb, caption="Selected face (blue box)", width=380)

st.divider()

# ── Step 3: process ────────────────────────────────────────────────────────────
st.subheader("Step 3 — Process the video")

if st.session_state.get("sel_box") is None:
    st.info("Complete Steps 1 and 2 first.")
    st.stop()

if st.button("Blur this face throughout the video", type="primary"):
    sf  = st.session_state.sel_frame
    sb  = st.session_state.sel_box

    tgt = get_embedding(embedder, frames[sf], sb)
    if tgt is None:
        st.error("Couldn't compute face embedding for the selected box. Try a different frame.")
        st.stop()

    prog = st.progress(0, text="Tracking forward…")
    fwd  = track_run(frames, list(range(sf, total)),       detector, embedder, sb, tgt)
    prog.progress(0.50, text="Tracking backward…")
    bwd  = track_run(frames, list(range(sf - 1, -1, -1)), detector, embedder, sb, tgt)
    prog.progress(0.85, text="Writing output…")

    merged  = {**fwd, **bwd}
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_tmp.close()

    writer = cv2.VideoWriter(
        out_tmp.name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (meta["width"], meta["height"]),
    )
    for i in range(total):
        writer.write(merged.get(i, frames[i]))
    writer.release()
    prog.progress(1.0, text="Done!")

    with open(out_tmp.name, "rb") as fh:
        st.download_button(
            "Download blurred video",
            fh,
            file_name=f"blurred_{Path(uploaded.name).stem}.mp4",
            mime="video/mp4",
        )
    os.unlink(out_tmp.name)
