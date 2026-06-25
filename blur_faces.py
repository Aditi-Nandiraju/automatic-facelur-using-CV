#!/usr/bin/env python3
"""
Face Tracking & Auto-Blur for Video Files
-------------------------------------------
Detects faces frame-by-frame using OpenCV's DNN face detector (ResNet-SSD,
res10_300x300) and blurs them, writing the result to a new video file.

Usage:
    python blur_faces.py --input input.mp4 --output output.mp4
    python blur_faces.py --input input.mp4 --output output.mp4 --confidence 0.6 --blur 35
    python blur_faces.py --input input.mp4 --output output.mp4 --mode pixelate
"""

import argparse
import os
import sys
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROTOTXT = os.path.join(SCRIPT_DIR, "models", "deploy.prototxt")
MODEL = os.path.join(SCRIPT_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")


def load_detector():
    if not os.path.exists(PROTOTXT) or not os.path.exists(MODEL):
        sys.exit(
            f"Model files not found. Expected:\n  {PROTOTXT}\n  {MODEL}\n"
            "Make sure the 'models' folder is alongside this script."
        )
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    return net


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
        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def blur_region(frame, box, mode="gaussian", strength=99):
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame

    if mode == "pixelate":
        # Shrink then enlarge to create a blocky pixelation effect
        ph, pw = roi.shape[:2]
        factor = max(1, strength // 10)
        small = cv2.resize(roi, (max(1, pw // factor), max(1, ph // factor)), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.resize(small, (pw, ph), interpolation=cv2.INTER_NEAREST)
    else:
        # Gaussian blur, kernel size must be odd and scale with box size
        k = strength
        if k % 2 == 0:
            k += 1
        k = max(5, min(k, 199))
        blurred = cv2.GaussianBlur(roi, (k, k), 0)

    frame[y1:y2, x1:x2] = blurred
    return frame


def expand_box(box, frame_shape, margin=0.15):
    """Slightly enlarge the box so blur fully covers the face edges."""
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


def process_video(input_path, output_path, confidence=0.5, mode="gaussian",
                   strength=99, margin=0.15, show_progress=True):
    net = load_detector()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        sys.exit(f"Could not open output video for writing: {output_path}")

    frame_idx = 0
    total_faces = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_faces(net, frame, confidence)
        for box in boxes:
            box = expand_box(box, frame.shape, margin)
            frame = blur_region(frame, box, mode=mode, strength=strength)
        total_faces += len(boxes)

        out.write(frame)
        frame_idx += 1

        if show_progress and total_frames > 0:
            pct = (frame_idx / total_frames) * 100
            print(f"\rProcessing: {frame_idx}/{total_frames} frames ({pct:.1f}%)", end="", flush=True)

    cap.release()
    out.release()
    if show_progress:
        print()
    print(f"Done. Wrote {frame_idx} frames, blurred {total_faces} face detections total.")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect and blur faces in a video file.")
    parser.add_argument("--input", "-i", required=True, help="Path to input video file")
    parser.add_argument("--output", "-o", required=True, help="Path to output video file")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                         help="Detection confidence threshold 0-1 (default 0.5)")
    parser.add_argument("--mode", "-m", choices=["gaussian", "pixelate"], default="gaussian",
                         help="Blur style (default gaussian)")
    parser.add_argument("--blur", "-b", type=int, default=99,
                         help="Blur strength: kernel size for gaussian, or block factor for pixelate (default 99)")
    parser.add_argument("--margin", type=float, default=0.15,
                         help="Fractional margin to expand each box by, helps fully cover faces (default 0.15)")
    args = parser.parse_args()

    process_video(
        args.input, args.output,
        confidence=args.confidence,
        mode=args.mode,
        strength=args.blur,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()