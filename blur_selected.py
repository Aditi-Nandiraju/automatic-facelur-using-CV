#!/usr/bin/env python3
"""
Scrub-and-Select Face Blur for Video Files (with face recognition)
---------------------------------------------------------------------
Lets you scrub to ANY frame in the video (not just the first one), click
on a face there, and the program finds that same person in every other
frame of the video -- using a face embedding (recognition) model to
re-identify them, not just spatial tracking -- and blurs only them.

Usage:
    python blur_selected.py --input input.mp4 --output output.mp4

Controls in the scrub window:
    - Drag the trackbar (or use Left/Right arrow keys) to find a frame
      where the person you want is clearly visible
    - Click on their face
    - Press 'q' to quit without processing

Requires the models/ folder next to this script:
    - deploy.prototxt + res10_300x300_ssd_iter_140000.caffemodel  (face detector)
    - openface_nn4_small2.t7                                      (face embedding/recognition)
See README.md for details on both models.
"""

import argparse
import os
import sys
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROTOTXT = os.path.join(SCRIPT_DIR, "models", "deploy.prototxt")
DETECTOR_MODEL = os.path.join(SCRIPT_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
EMBEDDER_MODEL = os.path.join(SCRIPT_DIR, "models", "openface_nn4_small2.t7")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_detector():
    if not os.path.exists(PROTOTXT) or not os.path.exists(DETECTOR_MODEL):
        sys.exit(
            f"Detector model files not found. Expected:\n  {PROTOTXT}\n  {DETECTOR_MODEL}\n"
            "Make sure the 'models' folder is alongside this script."
        )
    return cv2.dnn.readNetFromCaffe(PROTOTXT, DETECTOR_MODEL)


def load_embedder():
    if not os.path.exists(EMBEDDER_MODEL):
        sys.exit(
            f"Face embedding model not found. Expected:\n  {EMBEDDER_MODEL}\n"
            "Make sure the 'models' folder is alongside this script."
        )
    return cv2.dnn.readNetFromTorch(EMBEDDER_MODEL)


# ---------------------------------------------------------------------------
# Detection + embedding
# ---------------------------------------------------------------------------

def detect_faces(net, frame, confidence_threshold=0.5):
    """Returns list of (x1, y1, x2, y2) boxes for detected faces."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1.0, size=(300, 300),
        mean=(104.0, 177.0, 123.0), swapRB=False, crop=False
    )
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < confidence_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def get_embedding(embedder, frame, box):
    """128-d L2-normalized embedding vector for the face in `box`."""
    x1, y1, x2, y2 = box
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, (96, 96))
    blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(blob)
    vec = embedder.forward().flatten()
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    return vec / norm


def cosine_sim(a, b):
    return float(np.dot(a, b))


def pick_box_for_point(boxes, point):
    """Box containing the clicked point, or closest-center box otherwise."""
    if not boxes:
        return None
    px, py = point
    containing = [b for b in boxes if b[0] <= px <= b[2] and b[1] <= py <= b[3]]
    if containing:
        return min(containing, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    def center_dist(b):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        return (cx - px) ** 2 + (cy - py) ** 2

    return min(boxes, key=center_dist)


def box_to_xywh(box):
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)


def xywh_to_box(xywh, frame_shape):
    x, y, w, h = [int(v) for v in xywh]
    fh, fw = frame_shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw - 1, x + w), min(fh - 1, y + h)
    return (x1, y1, x2, y2)


def make_tracker():
    # CSRT was removed in OpenCV 4.10+; fall through to available alternatives
    for name in ("TrackerCSRT_create", "TrackerMIL_create"):
        if hasattr(cv2, name):
            return getattr(cv2, name)()
    if hasattr(cv2, "legacy"):
        for name in ("TrackerCSRT_create", "TrackerMIL_create"):
            if hasattr(cv2.legacy, name):
                return getattr(cv2.legacy, name)()
    raise RuntimeError("No suitable OpenCV tracker found. Install opencv-contrib-python.")


def blur_region(frame, box, strength=99):
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame
    k = strength
    if k % 2 == 0:
        k += 1
    k = max(5, min(k, 199))
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return frame


def expand_box(box, frame_shape, margin=0.30):
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    return (
        max(0, x1 - mx),
        max(0, y1 - my),
        min(w - 1, x2 + mx),
        min(h - 1, y2 + my),
    )


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def propagate_box_optical_flow(prev_frame, curr_frame, box):
    """Shift box by the median optical-flow displacement of points inside it."""
    x1, y1, x2, y2 = box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    nx, ny = max(3, bw // 10), max(3, bh // 10)
    pts = np.array(
        [[xi, yi] for xi in np.linspace(x1 + 2, x2 - 2, nx)
                  for yi in np.linspace(y1 + 2, y2 - 2, ny)],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    if len(pts) < 4:
        return None

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        gray_prev, gray_curr, pts, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    good = status.ravel() == 1
    if good.sum() < 3:
        return None

    dx = float(np.median(next_pts[good, 0, 0] - pts[good, 0, 0]))
    dy = float(np.median(next_pts[good, 0, 1] - pts[good, 0, 1]))

    h, w = curr_frame.shape[:2]
    return (
        max(0, int(round(x1 + dx))),
        max(0, int(round(y1 + dy))),
        min(w - 1, int(round(x2 + dx))),
        min(h - 1, int(round(y2 + dy))),
    )


# ---------------------------------------------------------------------------
# Scrub-and-click selection UI
# ---------------------------------------------------------------------------

def scrub_and_select(frames, detector, confidence=0.5):
    """Lets the user drag a trackbar through `frames` and click a face.
    Returns (frame_idx, box) or (None, None) if the user quit."""
    total = len(frames)
    window = "Scrub to a frame, then click the face to blur"
    cv2.namedWindow(window)
    cv2.createTrackbar("Frame", window, 0, max(0, total - 1), lambda v: None)

    state = {"idx": 0, "boxes": [], "chosen": None}

    def redraw():
        idx = state["idx"]
        frame = frames[idx].copy()
        state["boxes"] = detect_faces(detector, frame, confidence)
        for b in state["boxes"]:
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"Frame {idx}/{total - 1}  |  drag bar or Left/Right arrows  |  click a face  |  'q' quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.imshow(window, frame)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            box = pick_box_for_point(state["boxes"], (x, y))
            if box is not None:
                state["chosen"] = (state["idx"], box)

    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        if state["chosen"] is not None:
            break
        key = cv2.waitKey(30) & 0xFF
        trackbar_pos = cv2.getTrackbarPos("Frame", window)
        if trackbar_pos != state["idx"]:
            state["idx"] = trackbar_pos
            redraw()
        if key == ord('q'):
            break
        elif key == 81 or key == ord('a'):  # left arrow / 'a'
            state["idx"] = max(0, state["idx"] - 1)
            cv2.setTrackbarPos("Frame", window, state["idx"])
            redraw()
        elif key == 83 or key == ord('d'):  # right arrow / 'd'
            state["idx"] = min(total - 1, state["idx"] + 1)
            cv2.setTrackbarPos("Frame", window, state["idx"])
            redraw()

    cv2.destroyWindow(window)
    if state["chosen"] is None:
        return None, None
    return state["chosen"]


# ---------------------------------------------------------------------------
# Recognition-assisted tracking over a contiguous run of frames
# ---------------------------------------------------------------------------

def track_run(frames, indices, detector, embedder, init_box, target_embedding,
              confidence=0.4, sim_threshold=0.45):
    """
    Detection-first tracking with optical-flow coasting.

    Every frame: detect faces, pick the best embedding match from a growing
    gallery. When no face matches (turned away, occluded), use sparse optical
    flow to move the box with the subject's head rather than freezing it.
    During coast, accept a spatially-close face with a relaxed threshold so
    re-acquisition happens as soon as the subject begins turning back.
    """
    results = {}
    if not indices:
        return results

    GALLERY_ADD_THRESHOLD = 0.55
    MAX_GALLERY = 12
    MAX_COAST = 35
    ALPHA = 0.55

    gallery = [target_embedding]
    smooth = list(map(float, init_box))
    current_box = init_box
    coast = 0

    for step, idx in enumerate(indices):
        frame = frames[idx].copy()

        if step == 0:
            smooth = list(map(float, init_box))
            current_box = init_box
            coast = 0
            emb = get_embedding(embedder, frame, init_box)
            if emb is not None:
                gallery = [emb]
            results[idx] = blur_region(frame, expand_box(current_box, frame.shape))
            continue

        candidates = detect_faces(detector, frame, confidence)

        # Score each candidate: gallery similarity + small proximity bonus.
        # The bonus (max 0.07, fading over 3 face-widths) acts as a soft
        # tiebreaker toward the last known position without hard IoU gates that
        # break after optical-flow drift.
        best_box, best_sim, best_emb = None, -1.0, None
        lx = (current_box[0] + current_box[2]) / 2
        ly = (current_box[1] + current_box[3]) / 2
        for cand in candidates:
            emb = get_embedding(embedder, frame, cand)
            if emb is None:
                continue
            sim = max(cosine_sim(emb, ref) for ref in gallery)
            cx = (cand[0] + cand[2]) / 2
            cy = (cand[1] + cand[3]) / 2
            face_w = max(1, cand[2] - cand[0])
            dist_fw = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5 / face_w
            sim += 0.07 * max(0.0, 1.0 - dist_fw / 3.0)
            if sim > best_sim:
                best_sim, best_box, best_emb = sim, cand, emb

        if best_box is not None and best_sim >= sim_threshold:
            if coast > 0:
                smooth = list(map(float, best_box))
            else:
                for i, v in enumerate(best_box):
                    smooth[i] = ALPHA * v + (1 - ALPHA) * smooth[i]
            current_box = tuple(int(round(v)) for v in smooth)
            coast = 0
            if best_emb is not None and best_sim >= GALLERY_ADD_THRESHOLD and len(gallery) < MAX_GALLERY:
                gallery.append(best_emb)
            results[idx] = blur_region(frame, expand_box(current_box, frame.shape))
        elif coast < MAX_COAST:
            coast += 1
            of_box = propagate_box_optical_flow(frames[indices[step - 1]], frames[idx], current_box)
            if of_box is not None:
                for i, v in enumerate(of_box):
                    smooth[i] = ALPHA * v + (1 - ALPHA) * smooth[i]
                current_box = tuple(int(round(v)) for v in smooth)
            results[idx] = blur_region(frame, expand_box(current_box, frame.shape))
        else:
            results[idx] = frame

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_video(input_path, output_path, confidence=0.4, sim_threshold=0.45):
    detector = load_detector()
    embedder = load_embedder()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Loading video into memory for scrubbing...")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        sys.exit("No frames read from input video.")
    total = len(frames)
    print(f"Loaded {total} frames ({width}x{height} @ {fps:.2f} fps).")

    sel_idx, sel_box = scrub_and_select(frames, detector, confidence)
    if sel_box is None:
        print("No face selected. Exiting without processing.")
        return

    target_embedding = get_embedding(embedder, frames[sel_idx], sel_box)
    if target_embedding is None:
        sys.exit("Could not compute an embedding for the selected face. Try a different frame/box.")

    print(f"Selected face at frame {sel_idx}. Tracking forward and backward through the video...")

    forward_indices = list(range(sel_idx, total))
    backward_indices = list(range(sel_idx - 1, -1, -1))

    forward_results = track_run(
        frames, forward_indices, detector, embedder, sel_box, target_embedding,
        confidence=confidence, sim_threshold=sim_threshold,
    )
    backward_results = track_run(
        frames, backward_indices, detector, embedder, sel_box, target_embedding,
        confidence=confidence, sim_threshold=sim_threshold,
    )

    all_results = {**forward_results, **backward_results}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        sys.exit(f"Could not open output video for writing: {output_path}")

    for idx in range(total):
        out.write(all_results.get(idx, frames[idx]))
        if total > 0:
            pct = ((idx + 1) / total) * 100
            print(f"\rWriting output: {idx + 1}/{total} frames ({pct:.1f}%)", end="", flush=True)

    out.release()
    print()
    print(f"Done. Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrub to any frame, click a face, and blur that person through the whole video."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input video file")
    parser.add_argument("--output", "-o", required=True, help="Path to output video file")
    parser.add_argument("--confidence", "-c", type=float, default=0.4,
                         help="Face detection confidence threshold 0-1 (default 0.4)")
    parser.add_argument("--similarity", "-s", type=float, default=0.45,
                         help="Face recognition match threshold, cosine similarity 0-1 (default 0.45). "
                              "Lower = more lenient matching, higher = stricter.")
    args = parser.parse_args()

    process_video(
        args.input, args.output,
        confidence=args.confidence,
        sim_threshold=args.similarity,
    )


if __name__ == "__main__":
    main()