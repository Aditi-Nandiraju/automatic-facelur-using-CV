import io
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blur_selected import get_embedding, load_detector, load_embedder, track_run

MAX_MB  = 500
FORMATS = {"mp4", "avi", "mov", "mkv", "webm"}

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
PROTOTXT   = os.path.join(MODELS_DIR, "deploy.prototxt")
CAFFE      = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading models…", flush=True)
detector = load_detector()
embedder = load_embedder()
print("Models ready.", flush=True)

# video_id -> { frames, fps, width, height, out }
STORE: Dict[str, dict] = {}


# ── Upload ────────────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in FORMATS:
        raise HTTPException(400, f"Unsupported format .{ext}. Supported: {', '.join(sorted(FORMATS))}")

    data = await file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File is {len(data)//1024//1024} MB — limit is {MAX_MB} MB")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
    tmp.write(data)
    tmp.close()

    cap    = cv2.VideoCapture(tmp.name)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    os.unlink(tmp.name)

    if not frames:
        raise HTTPException(422, "Could not decode any frames from the video.")

    vid = str(uuid.uuid4())
    STORE[vid] = {"frames": frames, "fps": fps, "width": width, "height": height, "out": None}

    return {
        "videoId":     vid,
        "totalFrames": len(frames),
        "fps":         round(fps, 3),
        "width":       width,
        "height":      height,
        "duration":    round(len(frames) / fps, 2),
    }


# ── Frame image ───────────────────────────────────────────────────────────────
@app.get("/api/frame/{vid}/{idx}")
def frame(vid: str, idx: int):
    _check(vid, idx)
    ok, buf = cv2.imencode(".jpg", STORE[vid]["frames"][idx], [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise HTTPException(500, "Frame encode failed")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


# ── Face boxes ────────────────────────────────────────────────────────────────
@app.get("/api/faces/{vid}/{idx}")
def faces(vid: str, idx: int):
    _check(vid, idx)
    try:
        frame  = STORE[vid]["frames"][idx]
        h, w   = frame.shape[:2]
        blob   = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        detector.setInput(blob)
        dets   = detector.forward()
        boxes  = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < 0.4:
                continue
            b = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = b.astype(int).tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
        return {"boxes": boxes}
    except Exception as exc:
        raise HTTPException(500, f"Detection error: {exc}")


# ── Process ───────────────────────────────────────────────────────────────────
class ProcessReq(BaseModel):
    videoId:  str
    frameIdx: int
    box:      List[int]


@app.post("/api/process")
def process(req: ProcessReq):
    e = STORE.get(req.videoId)
    if not e:
        raise HTTPException(404, "Video not found — please upload again")

    frames = e["frames"]
    sf     = req.frameIdx
    sb     = tuple(req.box)
    total  = len(frames)

    if sf < 0 or sf >= total:
        raise HTTPException(400, f"frameIdx {sf} out of range")

    try:
        tgt = get_embedding(embedder, frames[sf], sb)
    except Exception as exc:
        raise HTTPException(500, f"Embedding error: {exc}")
    if tgt is None:
        raise HTTPException(422, "Could not compute embedding — try a clearer frame or larger face box.")

    try:
        fwd = track_run(frames, list(range(sf, total)),        detector, embedder, sb, tgt)
        bwd = track_run(frames, list(range(sf - 1, -1, -1)),   detector, embedder, sb, tgt)
    except Exception as exc:
        raise HTTPException(500, f"Tracking error: {exc}")
    merged = {**fwd, **bwd}

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out.close()

    writer = None
    for fourcc_str in ("avc1", "mp4v", "XVID"):
        try:
            w = cv2.VideoWriter(
                out.name, cv2.VideoWriter_fourcc(*fourcc_str),
                e["fps"], (e["width"], e["height"]),
            )
            if w.isOpened():
                writer = w
                break
            w.release()
        except Exception:
            continue
    if writer is None:
        raise HTTPException(500, "VideoWriter: no working codec found.")

    try:
        for i in range(total):
            writer.write(merged.get(i, frames[i]))
    finally:
        writer.release()

    if e.get("out") and os.path.exists(e["out"]):
        os.unlink(e["out"])
    e["out"] = out.name
    return {"status": "done"}


# ── Download ──────────────────────────────────────────────────────────────────
@app.get("/api/download/{vid}")
def download(vid: str):
    e = STORE.get(vid)
    if not e:
        raise HTTPException(404, "Video not found")
    path = e.get("out")
    if not path or not os.path.exists(path):
        raise HTTPException(404, "No processed video yet — run /api/process first")
    return FileResponse(path, media_type="video/mp4", filename=f"blurred_{vid[:8]}.mp4")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _check(vid: str, idx: int):
    e = STORE.get(vid)
    if not e:
        raise HTTPException(404, "Video not found — please re-upload")
    if idx < 0 or idx >= len(e["frames"]):
        raise HTTPException(400, f"Frame {idx} out of range (0–{len(e['frames'])-1})")
