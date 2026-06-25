"""
Face Blur — Gradio UI
Run: python gradio_app.py
"""
import os
import sys
import tempfile

import cv2
import gradio as gr
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from blur_selected import get_embedding, load_detector, load_embedder, track_run

print("Loading models…", flush=True)
detector = load_detector()
embedder = load_embedder()
print("Models ready.", flush=True)

_cache = {}   # holds decoded frames for the current video


def _bgr_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _draw_boxes(frame, boxes, selected_idx=None):
    vis = frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = (0, 200, 255) if i == selected_idx else (0, 220, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        label = f"Face {i + 1}" + (" ✓" if i == selected_idx else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(vis, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return vis


def _detect_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    detector.setInput(blob)
    dets = detector.forward()
    boxes = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < 0.4:
            continue
        b = (dets[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int).tolist()
        x1, y1 = max(0, b[0]), max(0, b[1])
        x2, y2 = min(w - 1, b[2]), min(h - 1, b[3])
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return boxes


# ── Event handlers ────────────────────────────────────────────────────────────

def on_upload(video):
    if video is None:
        return gr.update(maximum=1, value=0), None, "Upload a video to begin.", [], gr.update(choices=[]), ""

    # Gradio 4+ passes a string path; older versions may pass a dict
    path = video if isinstance(video, str) else (video.get("name") or video.get("path") or str(video))
    print(f"[upload] path={path}", flush=True)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return gr.update(maximum=1, value=0), None, f"Could not open video: {path}", [], gr.update(choices=[]), ""

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    if not frames:
        return gr.update(maximum=1, value=0), None, "Could not decode any frames.", [], gr.update(choices=[]), ""

    h, w = frames[0].shape[:2]
    _cache.clear()
    _cache["frames"] = frames
    _cache["fps"]    = fps
    _cache["w"]      = w
    _cache["h"]      = h

    info = f"Loaded {len(frames)} frames · {w}×{h} · {fps:.1f} fps"
    print(f"[upload] {info}", flush=True)

    # Run detection on frame 0 immediately
    boxes0   = _detect_faces(frames[0])
    vis0     = _draw_boxes(frames[0], boxes0)
    choices0 = [f"Face {i + 1}" for i in range(len(boxes0))]
    status0  = f"{len(boxes0)} face(s) at frame 0." if boxes0 else "No faces at frame 0 — drag slider."

    return (
        gr.update(maximum=len(frames) - 1, value=0),
        _bgr_to_rgb(vis0),
        info,
        boxes0,
        gr.update(choices=choices0, value=choices0[0] if choices0 else None),
        status0,
    )


def on_slider(frame_num):
    """Show frame and detect faces automatically whenever the slider moves."""
    if "frames" not in _cache:
        return None, [], gr.update(choices=[], value=None), ""

    frames = _cache["frames"]
    idx    = max(0, min(int(frame_num), len(frames) - 1))
    frame  = frames[idx]

    boxes   = _detect_faces(frame)
    vis     = _draw_boxes(frame, boxes)
    choices = [f"Face {i + 1}" for i in range(len(boxes))]

    if boxes:
        status = f"{len(boxes)} face(s) detected at frame {idx}. Select one below."
    else:
        status = f"No faces detected at frame {idx} — try another frame."

    return (
        _bgr_to_rgb(vis),
        boxes,
        gr.update(choices=choices, value=choices[0] if choices else None),
        status,
    )


def on_process(frame_num, face_choice, boxes, progress=gr.Progress()):
    if "frames" not in _cache:
        return None, "Upload a video first."
    if not boxes:
        return None, "Detect faces first."
    if not face_choice:
        return None, "Select a face to blur."

    frames  = _cache["frames"]
    fps     = _cache["fps"]
    w, h    = _cache["w"], _cache["h"]

    face_idx = int(face_choice.split()[-1]) - 1
    sf       = max(0, min(int(frame_num), len(frames) - 1))
    sel_box  = tuple(boxes[face_idx])

    print(f"[process] frame={sf}, box={sel_box}", flush=True)

    progress(0, desc="Computing face embedding…")
    tgt = get_embedding(embedder, frames[sf], sel_box)
    if tgt is None:
        return None, "Could not embed the face. Pick a clearer, front-facing frame."

    n = len(frames)

    progress(0.05, desc="Tracking forward…")
    fwd = track_run(frames, list(range(sf, n)),          detector, embedder, sel_box, tgt)

    progress(0.50, desc="Tracking backward…")
    bwd = track_run(frames, list(range(sf - 1, -1, -1)), detector, embedder, sel_box, tgt)

    merged = {**fwd, **bwd}

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out.close()

    writer = None
    for fc in ("avc1", "mp4v", "XVID"):
        try:
            wr = cv2.VideoWriter(out.name, cv2.VideoWriter_fourcc(*fc), fps, (w, h))
            if wr.isOpened():
                writer = wr
                break
            wr.release()
        except Exception:
            pass
    if writer is None:
        return None, "VideoWriter failed — no working codec found."

    progress(0.95, desc="Writing output…")
    for i in range(n):
        writer.write(merged.get(i, frames[i]))
        if i % max(1, n // 40) == 0:
            progress(0.95 + 0.05 * i / n, desc=f"Writing frame {i + 1}/{n}…")
    writer.release()

    progress(1.0, desc="Done!")
    print(f"[process] done, output={out.name}", flush=True)
    return out.name, f"Done! Processed {n} frames."


# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Face Blur") as demo:
    gr.Markdown(
        "## Face Blur\n"
        "1. Upload video → 2. Drag slider to a clear frame → "
        "3. **Detect Faces** → 4. Pick a face → 5. **Blur Selected Face**\n\n"
        "_MP4 · AVI · MOV · MKV · WEBM — max 500 MB_"
    )

    boxes_state = gr.State([])

    with gr.Row():
        video_in = gr.Video(label="Upload video", sources=["upload"])
        info_box = gr.Textbox(label="Status", interactive=False, lines=2)

    frame_slider = gr.Slider(0, 1, value=0, step=1, label="Frame number")

    with gr.Row():
        frame_img  = gr.Image(label="Frame with detected faces", type="numpy")
        with gr.Column():
            detect_status = gr.Textbox(label="Detection result", interactive=False, lines=3)
            face_radio    = gr.Radio(choices=[], label="Select face to blur")
            process_btn   = gr.Button("Blur Selected Face", variant="primary")

    with gr.Row():
        out_video   = gr.Video(label="Blurred video")
        proc_status = gr.Textbox(label="Processing status", interactive=False, lines=2)

    # Wiring
    video_in.change(
        on_upload,
        inputs=[video_in],
        outputs=[frame_slider, frame_img, info_box, boxes_state, face_radio, detect_status],
    )
    frame_slider.change(
        on_slider,
        inputs=[frame_slider],
        outputs=[frame_img, boxes_state, face_radio, detect_status],
    )
    process_btn.click(
        on_process,
        inputs=[frame_slider, face_radio, boxes_state],
        outputs=[out_video, proc_status],
    )


if __name__ == "__main__":
    demo.launch()
