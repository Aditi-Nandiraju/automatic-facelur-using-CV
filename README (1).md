# Face Blur

Blur a specific person's face in a video. Select them by clicking their face — the tool tracks and blurs only that person through the entire clip, leaving everyone else untouched.

## Requirements

- Python 3.8 or newer
- The `models/` folder with all three model files (included in this repo)

## Setup (run once)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/faceblur.git
cd faceblur

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
.\.venv\Scripts\activate      # Windows
source .venv/bin/activate     # Mac / Linux

# 4. Install dependencies
pip install -r requirements.txt
```

## Web app (Gradio)

```bash
python gradio_app.py
```

Open the URL printed in the terminal (default: `http://127.0.0.1:7860`).

**How to use:**

1. **Upload** your video (MP4, AVI, MOV, MKV, WEBM — max 500 MB)
2. **Drag the slider** to a frame where the target face is clearly visible and front-facing — detected faces are shown with green boxes automatically
3. **Select the face** to blur using the radio buttons
4. Click **Blur Selected Face** and wait — a progress bar shows tracking and writing progress
5. **Download** the blurred video from the output panel

**Supported formats & limits:**

| | |
|---|---|
| Video formats | MP4, AVI, MOV, MKV, WEBM |
| Max file size | 500 MB |
| RAM usage | Full video decoded into memory — expect 1–4 GB for a typical 100 MB clip |
| Speed | CPU only — a 30-second 720p clip takes ~1–3 minutes |

## CLI

### `blur_selected.py` — blur one specific person

```bash
python blur_selected.py --input input.mp4 --output output.mp4
```

A window opens. Drag the trackbar to find a frame with a clear view of the face, then click it. Processing runs automatically and saves the output.

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--confidence` | float | `0.4` | Face detection threshold, 0–1 |
| `--similarity` | float | `0.45` | Identity match threshold, 0–1 |

### `blur_faces.py` — blur every face in the video

```bash
python blur_faces.py --input input.mp4 --output output.mp4
```

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--confidence` | float | `0.5` | Detection threshold |
| `--mode` | str | `gaussian` | `"gaussian"` or `"pixelate"` |
| `--blur` | int | `99` | Blur strength |
| `--margin` | float | `0.15` | Expand box by this fraction |

## How it works

**Face detection** — every frame passes through a ResNet-SSD detector (`res10_300x300_ssd_iter_140000.caffemodel`), returning bounding boxes with confidence scores.

**Face recognition** — the selected face is embedded into a 128-d vector using OpenFace (`openface_nn4_small2.t7`). Each subsequent frame detects all faces, computes embeddings, and picks the one with the highest cosine similarity to a growing gallery of confirmed matches (up to 12 entries, threshold `0.55`). This lets the tracker adapt as pose and lighting change.

**Optical-flow coasting** — when the face turns away or is briefly occluded, Lucas-Kanade sparse optical flow shifts the last known box with the head's motion for up to ~1 second (30 frames), keeping the blur region on the subject without a fresh detection.

**Box smoothing** — bounding box position is blended with an exponential moving average (α = 0.55) between frames. On re-acquisition after coasting, the box snaps immediately.

## Notes

- **Audio is not preserved.** OpenCV's `VideoWriter` drops the audio track. To restore it:
  ```bash
  ffmpeg -i blurred.mp4 -i original.mp4 -c copy -map 0:v:0 -map 1:a:0 output_with_audio.mp4
  ```
- **Profile faces** may not be detected — the SSD model is trained on frontal faces. Optical-flow coasting bridges short gaps.
- **Accuracy is not 100%** — in crowded scenes or with fast motion, the tracker may occasionally blur the wrong face or miss frames.
