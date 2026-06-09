# Kriti 2026 - Keus Bot Vision Pipeline

A multi-model image recognition pipeline built for the Kriti 2026 robotics competition. The system detects and classifies objects placed on A4 sheets, including vehicles, pets, furniture, brand logos, smart switches, faces, QR codes, and number plates.

## Pipeline Architecture

```
Frame Selection (sharpness) → A4 Detection → Number OCR (PaddleOCR)
→ YOLO Detection (vehicles, pets, furniture)
→ Embedding Match (logos, switches)
→ Face Recognition (known faces)
→ QR Decode (pyzbar)
→ Number Plate OCR (PaddleOCR)
→ Merge Results
```

## Two Variants

### `recognize_mobilenet.py` — MobileNetV3 Edition
- Uses **MobileNetV3-Small** (torchvision) for embedding-based logo/switch matching
- 576-dimensional embeddings
- Requires a pre-built reference database (`db/`) via `--build-db`
- Lighter and faster — better suited for Raspberry Pi

### `recognize_clip.py` — CLIP Edition
- Uses **CLIP ViT-B/32** (open_clip) for both zero-shot classification and embedding matching
- 512-dimensional embeddings
- **Zero-shot capability**: can identify logos (Apple, Tesla, Maybach, Keus), smart switches, and parcels **without** a reference database
- Also supports database matching (`db_clip/`) for improved accuracy
- Larger model — better accuracy but slower inference

### Shared Components (identical in both)
- **YOLOv8-nano**: Object detection (vehicles, pets, furniture, persons)
- **PaddleOCR**: Image number reading and number plate OCR
- **face_recognition**: Identifies Henry Cavill, Keanu Reeves, Roger Federer
- **pyzbar**: QR code decoding
- **A4 sheet detection**: Perspective correction via contour detection
- **Frame selector**: Picks the sharpest frame from a rolling buffer (camera mode)

## Installation

```bash
# System dependency for QR decoding
sudo apt-get install -y libzbar0

# Python dependencies
pip install -r requirements.txt
```

> **Note:** `open_clip_torch` is only required for `recognize_clip.py`. If you only use `recognize_mobilenet.py`, you can skip it.

## Usage

### Build the reference database

```bash
# MobileNet version
python recognize_mobilenet.py --build-db --dataset ./dataset

# CLIP version
python recognize_clip.py --build-db --dataset ./dataset
```

### Recognize a single image

```bash
python recognize_clip.py --image photo.jpg
python recognize_mobilenet.py --image photo.jpg
```

### Test on a folder of images

```bash
python recognize_clip.py --test-folder ./test/
python recognize_mobilenet.py --test-folder ./test/
```

### Live camera modes

```bash
# Manual capture — press SPACE to recognize, Q to quit
python recognize_clip.py --camera

# Auto mode — continuously recognizes the sharpest frame
python recognize_clip.py --camera-auto
```

## Dataset Structure

The `--build-db` command expects a dataset organized as:

```
dataset/
├── Faces/
│   ├── Henry_Cavill/
│   ├── Keanu_Reeves/
│   └── Roger_Federer/
├── Logos/
│   ├── Apple/
│   ├── Keus/
│   ├── Maybach/
│   └── Tesla/
├── Parcel_box/
└── Switches/
    └── Smart_Console/
```

## Output Format

```
Image Number: 42
Category: Brand logo
Content: Apple
Inference: 850ms
A4 detected: True | Sharpness: 120.3
```
