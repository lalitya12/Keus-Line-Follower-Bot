#!/usr/bin/env python3
"""
Kriti 2026 - Keus Bot Vision Pipeline v2
==========================================

Architecture:
  Frame Selection (sharpness) → A4 Detection → Number OCR (PaddleOCR)
  → YOLO Detection (vehicles, pets, furniture, parcels)
  → Embedding Match (logos, switches)
  → Face Recognition (3 known faces)
  → QR Decode (pyzbar)
  → Number Plate OCR (PaddleOCR on plate region)
  → Merge Results

Target: ~1-2 seconds per image on Raspberry Pi 4GB

USAGE:
  python recognize.py --image photo.jpg                    # Single image
  python recognize.py --test-folder ./test/                # Test folder
  python recognize.py --build-db --dataset ./imageset      # Build embedding DB
  python recognize.py --camera                             # Live camera
  python recognize.py --camera-auto                        # Auto-detect + recognize

INSTALL (Raspberry Pi):
  sudo apt-get install -y libzbar0
  pip install ultralytics paddleocr paddlepaddle torch torchvision \\
      opencv-python-headless Pillow numpy pyzbar face_recognition \\
      --break-system-packages

INSTALL (WSL/Laptop for testing):
  pip install ultralytics paddleocr paddlepaddle torch torchvision \\
      opencv-python-headless Pillow numpy pyzbar
"""

import argparse
import json
import os
import pickle
import socket
import sys
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Suppress noisy logs from dependencies
logging.getLogger("ppocr").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Competition items (exhaustive list) ──
# YOLO COCO classes that map to competition categories
YOLO_TO_COMPETITION = {
    # Vehicles
    "car": ("Vehicle", "Car"),
    "truck": ("Vehicle", "Car"),       # trucks count as vehicles
    "bus": ("Vehicle", "Bus"),
    "bicycle": ("Vehicle", "Bicycle"),
    "motorcycle": ("Vehicle", "Motorbike"),
    # Pets
    "cat": ("Pets", "Cat"),
    "dog": ("Pets", "Dog"),
    # Furniture
    "chair": ("Furniture", "Chair"),
    "dining table": ("Furniture", "Dining Table"),
    "couch": ("Furniture", "Sofa"),
    "bed": ("Furniture", "Bed"),
    # Parcel-adjacent (YOLO doesn't have "parcel" class, handled separately)
    "suitcase": ("Parcel", "Parcel"),
    # Person (triggers face recognition)
    "person": None,  # Handled specially — triggers face detector
}

# YOLO confidence threshold
YOLO_CONFIDENCE = 0.30

# Dataset folder → category mapping
FOLDER_TO_CATEGORY = {
    "Furniture":       "Furniture",
    "Logos":           "Brand logo",
    "Number plate":    "Vehicle number plate",
    "Parcel box":      "Parcel",
    "Pets":            "Pets",
    "QR code":         "QR code",
    "Switches":        "Smart switch",
    "Vehicles":        "Vehicle",
    "Brand logo":      "Brand logo",
    "Face":            "Face recognition",
    "Faces":           "Face recognition",
    "face_recognition":"Face recognition",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ── Thresholds ──
EMBEDDING_THRESHOLD = 0.50          # cosine similarity for logo/switch match
FACE_DISTANCE_THRESHOLD = 0.55      # face_recognition (lower = stricter)
OCR_CONFIDENCE = 0.3                # PaddleOCR text confidence
PLATE_MIN_LEN = 5
PLATE_MAX_LEN = 15
SHARPNESS_THRESHOLD = 50.0          # Laplacian variance — below = too blurry

# Paths
DB_DIR = Path(__file__).parent / "db"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA STRUCTURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Detection:
    category: str
    content: str
    confidence: float
    method: str   # "yolo", "embedding", "qr", "plate_ocr", "face"
    bbox: tuple = None  # (x1, y1, x2, y2) if available


@dataclass
class RecognitionResult:
    image_number: str = ""
    detections: list = field(default_factory=list)
    inference_time_ms: float = 0.0
    sharpness: float = 0.0
    a4_detected: bool = False
    timings: dict = field(default_factory=dict)

    def add(self, category, content, confidence, method, bbox=None):
        self.detections.append(Detection(category, content, confidence, method, bbox))

    def format_output(self) -> str:
        lines = []

        if not self.detections:
            lines.append("Category: Unknown")
            lines.append("Content: Unknown")
        else:
            for i, d in enumerate(self.detections):
                lines.append(f"Category: {d.category}")
                lines.append(f"Content: {d.content}")
                if i < len(self.detections) - 1:
                    lines.append("")

        return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FRAME SELECTION (handles motion blur)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_sharpness(frame: np.ndarray) -> float:
    """
    Compute sharpness using Laplacian variance.
    Higher = sharper. Takes ~1-2ms per frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


class FrameSelector:
    """
    Keeps a rolling buffer of frames and picks the sharpest one.
    Call add_frame() on every camera frame.
    Call get_best_frame() when you want to recognize.
    """

    def __init__(self, buffer_size: int = 10):
        self.buffer_size = buffer_size
        self.frames = []       # (frame, sharpness)

    def add_frame(self, frame: np.ndarray) -> float:
        """Add frame to buffer, return its sharpness score."""
        score = compute_sharpness(frame)
        self.frames.append((frame.copy(), score))
        if len(self.frames) > self.buffer_size:
            self.frames.pop(0)
        return score

    def get_best_frame(self) -> tuple[np.ndarray | None, float]:
        """Return the sharpest frame in the buffer."""
        if not self.frames:
            return None, 0.0
        best = max(self.frames, key=lambda x: x[1])
        return best

    def clear(self):
        self.frames.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A4 SHEET DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_a4_sheet(img: np.ndarray) -> np.ndarray | None:
    """Detect A4 sheet, return perspective-corrected crop."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Try Canny
    edges = cv2.Canny(blurred, 40, 130)
    edges = cv2.dilate(edges, kernel, iterations=2)
    result = _find_sheet_contour(edges, img, h * w)
    if result is not None:
        return result

    # Try white detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 50, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    result = _find_sheet_contour(mask, img, h * w)
    if result is not None:
        return result

    return None


def _find_sheet_contour(binary, original, frame_area):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.08 or area > frame_area * 0.95:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4 and area > best_area:
            pts = approx.reshape(4, 2).astype(np.float32)

            # Aspect ratio check: A4 is ~1:1.414.
            # Accept ratios between 1:1.1 and 1:2.0 (portrait or landscape)
            widths = [np.linalg.norm(pts[i] - pts[(i+1) % 4]) for i in range(4)]
            widths.sort()
            short = (widths[0] + widths[1]) / 2
            long = (widths[2] + widths[3]) / 2
            if short < 1:
                continue
            ratio = long / short
            if ratio < 1.1 or ratio > 2.0:
                continue

            # Whiteness check: the region inside should be mostly white/light
            mask = np.zeros(original.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, approx.reshape(4, 2).astype(int), 255)
            region = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            mean_brightness = cv2.mean(region, mask=mask)[0]
            if mean_brightness < 150:  # A4 paper is white — should be bright
                continue

            best = pts
            best_area = area

    if best is None:
        return None

    rect = np.zeros((4, 2), dtype=np.float32)
    s = best.sum(axis=1)
    d = np.diff(best, axis=1).flatten()
    rect[0] = best[np.argmin(s)]
    rect[2] = best[np.argmax(s)]
    rect[1] = best[np.argmin(d)]
    rect[3] = best[np.argmax(d)]

    out_w, out_h = 400, 566
    dst = np.array([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(original, M, (out_w, out_h))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# YOLO DETECTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class YOLODetector:
    """YOLOv8-nano for fast object detection."""

    def __init__(self):
        from ultralytics import YOLO
        # YOLOv8-nano: smallest, fastest. ~6MB model.
        # Auto-downloads on first run.
        self.model = YOLO("yolov8n.pt")
        self.model.fuse()  # Fuse layers for faster inference
        print("  YOLOv8-nano loaded")

    def detect(self, img: np.ndarray) -> list[Detection]:
        """Run YOLO detection, return competition-mapped results."""
        results = self.model.predict(img, conf=YOLO_CONFIDENCE, verbose=False,
                                      imgsz=320)[0]

        detections = []
        person_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            bbox = (int(x1), int(y1), int(x2), int(y2))

            if cls_name == "person":
                person_boxes.append(bbox)
                continue

            mapping = YOLO_TO_COMPETITION.get(cls_name)
            if mapping:
                cat, content = mapping
                detections.append(Detection(cat, content, conf, "yolo", bbox))

        return detections, person_boxes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PADDLEOCR (for number + plate reading)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class OCREngine:
    """Tesseract for number reading, PaddleOCR for plate reading."""

    def __init__(self):
        # Lightweight Tesseract for number reading (~70ms vs ~1200ms PaddleOCR)
        try:
            import pytesseract
            pytesseract.image_to_string(np.zeros((10, 10), dtype=np.uint8))
            self.tesseract = pytesseract
            print("  Tesseract loaded (for number + plate OCR)")
        except Exception:
            self.tesseract = None
            print("  Tesseract not available")

        # PaddleOCR only as fallback if Tesseract unavailable
        self.ocr = None
        self.use_predict = False
        if not self.tesseract:
            try:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(lang="en")
                self.use_predict = hasattr(self.ocr, 'predict') and not hasattr(self.ocr, 'ocr')
                print("  PaddleOCR loaded (fallback for plate OCR)")
            except Exception:
                print("  PaddleOCR not available either")

    def _safe_ocr(self, img: np.ndarray) -> list[tuple[str, float]]:
        """
        Run OCR and return [(text, confidence), ...].
        Handles both old (.ocr()) and new (.predict()) PaddleOCR APIs.
        """
        texts = []
        if self.ocr is None:
            return texts
        result = None

        try:
            # Try .predict() first (new API), then .ocr() (old API)
            if self.use_predict:
                result = self.ocr.predict(img)
            else:
                try:
                    result = self.ocr.ocr(img)
                except Exception:
                    try:
                        result = self.ocr.predict(img)
                    except Exception:
                        return texts
        except Exception:
            return texts

        if not result:
            return texts

        # Handle multiple possible result formats:
        # Old format: [[  [bbox, (text, conf)], ...  ]]
        # New format: [{"rec_texts": [...], "rec_scores": [...]}]
        # Or: list of dicts with text/score keys

        try:
            # New predict() format: list of dicts
            if isinstance(result, dict):
                result = [result]

            if isinstance(result, list) and result:
                first = result[0]

                # Format: dict with rec_texts / rec_scores
                if isinstance(first, dict):
                    rec_texts = first.get("rec_texts", first.get("texts", []))
                    rec_scores = first.get("rec_scores", first.get("scores", []))
                    if rec_texts:
                        for t, s in zip(rec_texts, rec_scores):
                            texts.append((str(t).strip(), float(s)))
                        return texts

                # Format: [[bbox, (text, conf)], ...]  or  [None]
                lines = result
                if isinstance(first, list) and first and isinstance(first[0], (list, np.ndarray)):
                    # Double-nested: result[0] is the actual lines
                    lines = first

                for line in lines:
                    if line is None:
                        continue
                    try:
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            text_part = line[1]
                            if isinstance(text_part, (list, tuple)) and len(text_part) >= 2:
                                texts.append((str(text_part[0]).strip(),
                                              float(text_part[1])))
                    except (IndexError, TypeError, ValueError):
                        continue
        except Exception:
            pass

        return texts

    def _fast_rec(self, img: np.ndarray) -> list[tuple[str, float]]:
        """Recognition-only OCR (no text detection) — much faster for known regions."""
        return self._safe_ocr(img)

    def read_number(self, region: np.ndarray) -> str:
        """Read the image number from the top-left corner using Tesseract."""
        if region.size == 0:
            return ""

        try:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            scaled = cv2.resize(thresh, None, fx=2, fy=2,
                                interpolation=cv2.INTER_CUBIC)

            if self.tesseract:
                # Tesseract: ~70ms vs PaddleOCR ~1200ms
                text = self.tesseract.image_to_string(
                    scaled, config='--psm 7 digits'
                ).strip()
                digits = "".join(c for c in text if c.isdigit())
                if digits:
                    return digits
            else:
                # Fallback to PaddleOCR
                scaled_bgr = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
                for text, conf in self._safe_ocr(scaled_bgr):
                    digits = "".join(c for c in text if c.isdigit())
                    if digits and conf >= OCR_CONFIDENCE:
                        return digits
        except Exception as e:
            print(f"  [OCR] Number error: {e}")
        return ""

    def read_plate(self, region: np.ndarray) -> str | None:
        """Read number plate text from a cropped region using Tesseract."""
        if region.size == 0:
            return None

        try:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            if self.tesseract:
                text = self.tesseract.image_to_string(
                    scaled, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                ).strip().upper().replace(" ", "")
            else:
                # Fallback to PaddleOCR
                texts = []
                scaled_bgr = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
                for t, conf in self._safe_ocr(scaled_bgr):
                    if conf >= OCR_CONFIDENCE:
                        texts.append(t)
                text = "".join(texts).replace(" ", "").upper()

            has_letters = any(c.isalpha() for c in text)
            has_digits = any(c.isdigit() for c in text)
            valid_len = PLATE_MIN_LEN <= len(text) <= PLATE_MAX_LEN

            if has_letters and has_digits and valid_len:
                return text
        except Exception as e:
            print(f"  [OCR] Plate error: {e}")
        return None

    def detect_plate_in_image(self, img: np.ndarray) -> Detection | None:
        """Try to find and read a number plate in the full image using Tesseract."""
        if img.size == 0:
            return None

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.tesseract:
                text = self.tesseract.image_to_string(
                    gray, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                ).strip()
                # Try to find plate-like patterns in all detected lines
                for line in text.split('\n'):
                    cleaned = line.replace(" ", "").upper()
                    has_letters = any(c.isalpha() for c in cleaned)
                    has_digits = any(c.isdigit() for c in cleaned)
                    alnum = sum(c.isalnum() for c in cleaned)
                    ratio = alnum / max(len(cleaned), 1)
                    valid_len = PLATE_MIN_LEN <= len(cleaned) <= PLATE_MAX_LEN

                    if has_letters and has_digits and valid_len and ratio >= 0.8:
                        return Detection("Vehicle number plate", cleaned,
                                         0.85, "plate_ocr")
            else:
                # Fallback to PaddleOCR
                for t, conf in self._safe_ocr(img):
                    cleaned = t.replace(" ", "").upper()
                    has_letters = any(c.isalpha() for c in cleaned)
                    has_digits = any(c.isdigit() for c in cleaned)
                    alnum = sum(c.isalnum() for c in cleaned)
                    ratio = alnum / max(len(cleaned), 1)
                    valid_len = PLATE_MIN_LEN <= len(cleaned) <= PLATE_MAX_LEN

                    if (has_letters and has_digits and valid_len and
                        ratio >= 0.8 and conf >= OCR_CONFIDENCE):
                        return Detection("Vehicle number plate", cleaned,
                                         conf, "plate_ocr")
        except Exception as e:
            print(f"  [OCR] Plate detect error: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMBEDDING MATCHER (for logos + switches)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EmbeddingMatcher:
    """
    MobileNetV3-Small embeddings for matching brand logos and switches
    against the reference dataset.
    """

    def __init__(self, db_dir: str = str(DB_DIR)):
        import torch
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as T

        self.torch = torch

        # Check for fine-tuned weights first, fall back to pretrained ImageNet
        finetuned_path = Path(__file__).parent / "mobilenet_finetuned.pth"
        if finetuned_path.exists():
            checkpoint = torch.load(str(finetuned_path), map_location="cpu",
                                    weights_only=False)
            num_classes = checkpoint.get("num_classes", 10)
            self.model = models.mobilenet_v3_small(weights=None)
            self.model.classifier = nn.Sequential(
                nn.Linear(576, 256),
                nn.Hardswish(),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_classes),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            # Replace classifier with Identity for embedding extraction
            self.model.classifier = nn.Identity()
            print(f"  Loaded FINE-TUNED MobileNetV3 (val acc: "
                  f"{checkpoint.get('best_val_acc', 0):.1%})")
        else:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.model = models.mobilenet_v3_small(weights=weights)
            self.model.classifier = nn.Identity()
            print("  Using pretrained ImageNet MobileNetV3 (no fine-tuned weights found)")
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # Load DB
        self.has_db = False
        emb_file = Path(db_dir) / "embeddings.pkl"
        if emb_file.exists():
            with open(emb_file, "rb") as f:
                data = pickle.load(f)
            self.db_embeddings = data["embeddings"]
            self.db_metadata = data["metadata"]
            self.has_db = True
            print(f"  Embedding DB: {len(self.db_metadata)} references")
        else:
            self.db_embeddings = np.array([])
            self.db_metadata = []

        print("  MobileNetV3 embeddings ready")

    def match(self, img: np.ndarray) -> list[Detection]:
        """Find closest matches from the reference DB."""
        if not self.has_db or len(self.db_embeddings) == 0:
            return []

        query = self._embed(img)
        sims = self.db_embeddings @ query
        top_idx = np.argsort(sims)[::-1]

        dets, seen = [], set()
        for idx in top_idx:
            sim = float(sims[idx])
            if sim < EMBEDDING_THRESHOLD:
                break
            m = self.db_metadata[idx]
            key = (m["category"], m["content"])
            if key not in seen:
                seen.add(key)
                dets.append(Detection(m["category"], m["content"],
                                      sim, "embedding"))
            if len(dets) >= 3:  # Max 3 embedding matches
                break
        return dets

    def _embed(self, img: np.ndarray) -> np.ndarray:
        """Extract 576-dim normalized embedding from BGR image."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        with self.torch.no_grad():
            tensor = self.transform(pil).unsqueeze(0)
            feat = self.model(tensor).squeeze().cpu().numpy()
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FACE RECOGNITION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FaceRecognizer:
    """Recognize Henry Cavill, Keanu Reeves, Roger Federer."""

    def __init__(self, db_dir: str = str(DB_DIR)):
        self.available = False

        try:
            import face_recognition
            self.fr = face_recognition
        except ImportError:
            print("  Face recognition: not installed")
            return

        face_file = Path(db_dir) / "face_encodings.pkl"
        if face_file.exists():
            with open(face_file, "rb") as f:
                data = pickle.load(f)
            if data["encodings"]:
                self.encodings = np.array(data["encodings"])
                self.names = data["names"]
                self.available = True
                print(f"  Face DB: {len(self.names)} faces")
                return

        print("  Face DB: not found (run --build-db with face images)")

    def recognize(self, img: np.ndarray, person_boxes: list = None) -> list[Detection]:
        """
        Recognize faces in image.
        If person_boxes provided (from YOLO), crop to those regions first.
        """
        if not self.available:
            return []

        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if person_boxes:
                # Only look for faces in YOLO person regions
                all_dets = []
                for (x1, y1, x2, y2) in person_boxes:
                    # Expand box slightly
                    h, w = img.shape[:2]
                    pad = 20
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(w, x2 + pad)
                    y2 = min(h, y2 + pad)

                    crop = rgb[y1:y2, x1:x2].copy()
                    if crop.size == 0:
                        continue

                    # Downscale crop for faster HOG face detection
                    max_w = 500
                    ch, cw = crop.shape[:2]
                    if cw > max_w:
                        scale = max_w / cw
                        small = cv2.resize(crop, (max_w, int(ch * scale)))
                    else:
                        small = crop
                        scale = 1.0

                    locs = self.fr.face_locations(small, model="hog")
                    if not locs:
                        continue

                    # Scale locations back to original crop size for encoding
                    if scale != 1.0:
                        locs = [(int(t/scale), int(r/scale), int(b/scale), int(l/scale))
                                for t, r, b, l in locs]

                    encs = self.fr.face_encodings(crop, locs)
                    for enc in encs:
                        dists = self.fr.face_distance(self.encodings, enc)
                        best = int(np.argmin(dists))
                        if dists[best] <= FACE_DISTANCE_THRESHOLD:
                            all_dets.append(Detection(
                                "Face recognition",
                                self.names[best],
                                1.0 - float(dists[best]),
                                "face"
                            ))
                return all_dets
            else:
                # Search whole image — downscale for faster HOG
                max_w = 500
                h, w = rgb.shape[:2]
                if w > max_w:
                    scale = max_w / w
                    small = cv2.resize(rgb, (max_w, int(h * scale)))
                else:
                    small = rgb
                    scale = 1.0

                locs = self.fr.face_locations(small, model="hog")
                if not locs:
                    return []

                # Scale back to original for encoding
                if scale != 1.0:
                    locs = [(int(t/scale), int(r/scale), int(b/scale), int(l/scale))
                            for t, r, b, l in locs]

                encs = self.fr.face_encodings(rgb, locs)
                dets = []
                for enc in encs:
                    dists = self.fr.face_distance(self.encodings, enc)
                    best = int(np.argmin(dists))
                    if dists[best] <= FACE_DISTANCE_THRESHOLD:
                        dets.append(Detection(
                            "Face recognition",
                            self.names[best],
                            1.0 - float(dists[best]),
                            "face"
                        ))
                return dets
        except Exception as e:
            print(f"  [Face] Error: {e}")
            return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QR DECODER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class QRDecoder:
    def __init__(self):
        self.available = False
        try:
            from pyzbar.pyzbar import decode
            self.decode = decode
            self.available = True
            print("  QR decoder: ready")
        except ImportError:
            print("  QR decoder: pyzbar not installed")

    def detect(self, img: np.ndarray) -> Detection | None:
        if not self.available or img.size == 0:
            return None
        try:
            # Try original
            decoded = self.decode(img)
            if decoded:
                return Detection("QR code",
                                 decoded[0].data.decode("utf-8", errors="replace"),
                                 1.0, "qr")
            # Try grayscale + threshold
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for thresh_img in [gray,
                                cv2.threshold(gray, 0, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]]:
                decoded = self.decode(thresh_img)
                if decoded:
                    return Detection("QR code",
                                     decoded[0].data.decode("utf-8", errors="replace"),
                                     1.0, "qr")
        except Exception as e:
            print(f"  [QR] Error: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN RECOGNIZER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ImageRecognizer:
    """
    Main pipeline orchestrator.

    Runs detectors in smart order:
      1. YOLO first (fast, catches vehicles/pets/furniture)
      2. QR decoder (instant)
      3. Based on YOLO results, selectively run:
         - Face recognition (only if YOLO found a person)
         - Plate OCR (only if YOLO found a vehicle)
      4. Embedding match (for logos/switches not covered by YOLO)
    """

    def __init__(self, db_dir: str = str(DB_DIR)):
        print("\n" + "=" * 50)
        print("INITIALIZING RECOGNITION PIPELINE")
        print("=" * 50)

        t0 = time.time()
        self.yolo = YOLODetector()
        self.embedder = EmbeddingMatcher(db_dir)
        self.faces = FaceRecognizer(db_dir)
        self.qr = QRDecoder()
        self.ocr = OCREngine()  # Loaded last — slowest init

        print(f"\nAll models loaded in {time.time()-t0:.1f}s")
        print("=" * 50 + "\n")

    def recognize(self, image_input) -> RecognitionResult:
        """
        Full recognition pipeline.
        Input: file path or BGR numpy array.
        """
        start = time.time()
        result = RecognitionResult()
        timings = {}

        # ── Load image ──
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                print(f"ERROR: Cannot read {image_input}")
                return result
        else:
            img = image_input

        result.sharpness = compute_sharpness(img)

        # ── A4 Detection ──
        t = time.time()
        sheet = detect_a4_sheet(img)
        timings["a4_detect"] = (time.time() - t) * 1000

        if sheet is not None:
            result.a4_detected = True
            h, w = sheet.shape[:2]
            number_region = sheet[0:int(h * 0.14), 0:int(w * 0.22)]
            content = sheet[int(h * 0.10):int(h * 0.95),
                           int(w * 0.05):int(w * 0.95)]
        else:
            result.a4_detected = False
            h, w = img.shape[:2]
            number_region = img[0:int(h * 0.12), 0:int(w * 0.18)]
            content = img

        # ── Resize content to max 640px wide for faster downstream processing ──
        ch, cw = content.shape[:2]
        MAX_CONTENT_W = 640
        content_scale = 1.0
        if cw > MAX_CONTENT_W:
            content_scale = MAX_CONTENT_W / cw
            content = cv2.resize(content, (MAX_CONTENT_W, int(ch * content_scale)))

        # ── Number OCR (on small region only — fast) ──
        t = time.time()
        result.image_number = self.ocr.read_number(number_region)
        timings["number_ocr"] = (time.time() - t) * 1000

        # ── YOLO Detection ──
        t = time.time()
        yolo_dets, person_boxes = self.yolo.detect(content)
        timings["yolo"] = (time.time() - t) * 1000

        # ── QR Code (instant) ──
        t = time.time()
        qr_det = self.qr.detect(content)
        if qr_det is None and not result.a4_detected:
            qr_det = self.qr.detect(img)
        timings["qr"] = (time.time() - t) * 1000

        # ── Conditional: Face Recognition ──
        # Run face detection if: YOLO found person, OR YOLO found nothing
        # (could be a close-up face that YOLO misses)
        face_dets = []
        face_attempted = False
        t = time.time()
        if person_boxes or not yolo_dets:
            face_attempted = True
            # Use original full-res image for face recognition (needs resolution
            # for accurate encoding); the face recognizer handles its own downscaling.
            # Scale person_boxes from content coords back to original image coords.
            scaled_boxes = None
            if person_boxes and content_scale != 1.0:
                inv = 1.0 / content_scale
                scaled_boxes = [(int(x1*inv), int(y1*inv), int(x2*inv), int(y2*inv))
                                for (x1, y1, x2, y2) in person_boxes]
            elif person_boxes:
                scaled_boxes = person_boxes
            face_dets = self.faces.recognize(img, scaled_boxes)
        timings["face"] = (time.time() - t) * 1000

        # If YOLO detected person but face_recognition isn't available,
        # still report "Face recognition" as category with unknown content
        if person_boxes and not face_dets and not self.faces.available:
            for bbox in person_boxes:
                result.add("Face recognition", "Unknown person", 0.5, "yolo", bbox)

        # ── Conditional: Plate OCR (only if vehicle detected) ──
        plate_det = None
        if self._has_vehicle(yolo_dets):
            t = time.time()
            plate_det = self.ocr.detect_plate_in_image(content)
            timings["plate_ocr"] = (time.time() - t) * 1000

        # ── Embedding Match (skip if YOLO/face already handled the image) ──
        # Don't run embedding matcher when face detection was attempted —
        # a face image matched against logos/parcels produces garbage results.
        emb_dets = []
        if not yolo_dets and not face_dets and not face_attempted:
            t = time.time()
            emb_dets = self.embedder.match(content)
            timings["embedding"] = (time.time() - t) * 1000
        else:
            timings["embedding"] = 0.0

        # ── Merge all results ──
        self._merge(result, yolo_dets, emb_dets, qr_det,
                    plate_det, face_dets)

        result.timings = timings
        result.inference_time_ms = (time.time() - start) * 1000
        return result

    def _has_vehicle(self, yolo_dets):
        return any(d.category == "Vehicle" for d in yolo_dets)

    def _merge(self, result, yolo_dets, emb_dets, qr_det,
               plate_det, face_dets):
        """
        Merge detections with priority:
          QR > Face > Plate > YOLO > Embedding
        """
        all_dets = []

        # QR (highest priority — deterministic)
        if qr_det:
            all_dets.append(qr_det)

        # Face recognition
        for fd in face_dets:
            all_dets.append(fd)

        # Number plate
        if plate_det:
            all_dets.append(plate_det)

        # YOLO detections
        for yd in yolo_dets:
            all_dets.append(yd)

        # Embedding matches (only if not already covered)
        covered_cats = {d.category for d in all_dets}
        for ed in emb_dets:
            if ed.category not in covered_cats:
                all_dets.append(ed)

        # Sort by confidence, deduplicate
        all_dets.sort(key=lambda d: d.confidence, reverse=True)
        seen = set()
        for d in all_dets:
            key = (d.category, d.content)
            if key not in seen:
                seen.add(key)
                result.add(d.category, d.content, d.confidence, d.method)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATABASE BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_database(dataset_dirs, db_dir=str(DB_DIR)):
    """Build embedding + face database from dataset folders."""
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T

    output = Path(db_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BUILDING REFERENCE DATABASE")
    print("=" * 60)

    # Scan
    all_entries = []
    for ds in dataset_dirs:
        p = Path(ds)
        if not p.exists():
            print(f"WARNING: {p} not found")
            continue
        entries = _scan_dataset(p)
        all_entries.extend(entries)
        print(f"  {p.name}: {len(entries)} images")

    if not all_entries:
        print("ERROR: No images found!")
        sys.exit(1)

    cats = {}
    for e in all_entries:
        cats[e["category"]] = cats.get(e["category"], 0) + 1
    print(f"\nTotal: {len(all_entries)} images:")
    for c, n in sorted(cats.items()):
        print(f"  {c}: {n}")

    # Embeddings — use fine-tuned model if available
    finetuned_path = Path(__file__).parent / "mobilenet_finetuned.pth"
    if finetuned_path.exists():
        checkpoint = torch.load(str(finetuned_path), map_location="cpu",
                                weights_only=False)
        num_classes = checkpoint.get("num_classes", 10)
        model = models.mobilenet_v3_small(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.classifier = nn.Identity()
        print("  Using FINE-TUNED MobileNetV3 for embeddings")
    else:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier = nn.Identity()
        print("  Using pretrained ImageNet MobileNetV3 for embeddings")
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    embeddings, metadata = [], []
    # Skip face entries — faces are handled by dlib, not MobileNet
    non_face_entries = [e for e in all_entries
                        if "face" not in e["category"].lower()]
    print(f"\nComputing embeddings ({len(non_face_entries)} non-face images)...")
    for i, entry in enumerate(non_face_entries):
        try:
            img = Image.open(entry["path"]).convert("RGB")
            with torch.no_grad():
                feat = model(transform(img).unsqueeze(0)).squeeze().numpy()
            norm = np.linalg.norm(feat)
            embeddings.append(feat / norm if norm > 0 else feat)
            metadata.append(entry)
        except Exception as e:
            print(f"  FAILED: {entry['path']}: {e}")
        if (i + 1) % 20 == 0 or (i + 1) == len(non_face_entries):
            print(f"  {i+1}/{len(non_face_entries)}")

    with open(output / "embeddings.pkl", "wb") as f:
        pickle.dump({"embeddings": np.array(embeddings, dtype=np.float32),
                      "metadata": metadata}, f)
    print(f"Saved embeddings: ({len(embeddings)}, 576)")

    # Faces
    _build_faces(all_entries, output)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


def _scan_dataset(root):
    entries = []
    for cat_folder in sorted(root.iterdir()):
        if not cat_folder.is_dir():
            continue
        category = FOLDER_TO_CATEGORY.get(cat_folder.name, cat_folder.name)
        subfolders = [p for p in cat_folder.iterdir() if p.is_dir()]
        if subfolders:
            for sf in sorted(subfolders):
                content = sf.name.replace("_", " ")
                for img in sorted(sf.iterdir()):
                    if img.suffix.lower() in IMAGE_EXTENSIONS:
                        entries.append({"path": str(img), "category": category,
                                        "content": content})
        else:
            for img in sorted(cat_folder.iterdir()):
                if img.suffix.lower() in IMAGE_EXTENSIONS:
                    entries.append({"path": str(img), "category": category,
                                    "content": category})
    return entries


def _build_faces(entries, output):
    try:
        import face_recognition as fr
    except ImportError:
        print("\n[Face] not installed — skipping")
        with open(output / "face_encodings.pkl", "wb") as f:
            pickle.dump({"encodings": [], "names": []}, f)
        return

    face_entries = [e for e in entries if "face" in e["category"].lower()]
    encodings, names = [], []
    if face_entries:
        print(f"\n[Face] Encoding {len(face_entries)} face images...")
        for e in face_entries:
            try:
                img = fr.load_image_file(e["path"])
                encs = fr.face_encodings(img)
                if encs:
                    encodings.append(encs[0])
                    names.append(e["content"])
                    print(f"  {e['content']}")
            except Exception as ex:
                print(f"  FAILED: {e['path']}: {ex}")

    with open(output / "face_encodings.pkl", "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
    print(f"[Face] {len(encodings)} encodings saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PERSISTENT SERVER (load models once, accept images via socket)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_server(socket_path: str, db_dir: str):
    """
    Persistent recognition server.
    Loads all models once, then listens on a Unix socket for image paths.

    Protocol (newline-delimited JSON):
      Request:  {"image": "/path/to/image.jpg"}   or   {"command": "quit"}
      Response: {"detections": [...], "image_number": "3", "inference_ms": 450, ...}

    Usage from robot controller:
        import socket, json
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect("/tmp/kriti_vision.sock")
        sock.sendall(json.dumps({"image": "/path/to/photo.jpg"}).encode() + b"\\n")
        response = json.loads(sock.makefile().readline())
        print(response["detections"])
        sock.close()
    """
    # Remove stale socket
    if os.path.exists(socket_path):
        os.remove(socket_path)

    recognizer = ImageRecognizer(db_dir)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    print(f"\n[Server] Listening on {socket_path}")
    print("[Server] Models loaded — ready for requests")
    print("[Server] Send {{\"command\": \"quit\"}} to stop\n")

    try:
        while True:
            conn, _ = server.accept()
            try:
                data = conn.makefile().readline().strip()
                if not data:
                    conn.close()
                    continue

                req = json.loads(data)

                if req.get("command") == "quit":
                    conn.sendall(json.dumps({"status": "shutting_down"}).encode() + b"\n")
                    conn.close()
                    break

                image_path = req.get("image")
                if not image_path:
                    conn.sendall(json.dumps({"error": "missing 'image' key"}).encode() + b"\n")
                    conn.close()
                    continue

                result = recognizer.recognize(image_path)
                response = {
                    "detections": [
                        {"category": d.category, "content": d.content,
                         "confidence": round(d.confidence, 3), "source": d.source}
                        for d in result.detections
                    ],
                    "image_number": result.image_number,
                    "a4_detected": result.a4_detected,
                    "inference_ms": round(result.inference_time_ms, 1),
                    "timings": {k: round(v, 1) for k, v in result.timings.items()},
                }
                conn.sendall(json.dumps(response).encode() + b"\n")
                print(f"[Server] {Path(image_path).name}: {result.inference_time_ms:.0f}ms "
                      f"→ {[d.category + ':' + d.content for d in result.detections]}")

            except json.JSONDecodeError:
                conn.sendall(json.dumps({"error": "invalid JSON"}).encode() + b"\n")
            except Exception as e:
                conn.sendall(json.dumps({"error": str(e)}).encode() + b"\n")
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\n[Server] Stopped")
    finally:
        server.close()
        if os.path.exists(socket_path):
            os.remove(socket_path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="Kriti 2026 Vision Pipeline v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python recognize.py --image photo.jpg
  python recognize.py --test-folder ./test/
  python recognize.py --build-db --dataset ./imageset
  python recognize.py --camera
  python recognize.py --camera-auto
  python recognize.py --server                  # Start persistent server
        """
    )
    parser.add_argument("--build-db", action="store_true")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--image", type=str)
    parser.add_argument("--test-folder", type=str)
    parser.add_argument("--camera", action="store_true",
                        help="Manual camera — press SPACE to capture")
    parser.add_argument("--camera-auto", action="store_true",
                        help="Auto camera — recognizes sharpest frame continuously")
    parser.add_argument("--server", action="store_true",
                        help="Persistent server — loads once, accepts images via socket")
    parser.add_argument("--server-socket", type=str,
                        default="/tmp/kriti_vision.sock",
                        help="Unix socket path for --server mode")
    parser.add_argument("--db-dir", type=str, default=str(DB_DIR))
    args = parser.parse_args()

    if args.build_db:
        if not args.dataset:
            print("ERROR: --dataset required")
            sys.exit(1)
        build_database(args.dataset, args.db_dir)
        return

    if args.image:
        recognizer = ImageRecognizer(args.db_dir)
        result = recognizer.recognize(args.image)
        print("\n" + "─" * 50)
        print(result.format_output())
        print("─" * 50)
        return

    if args.test_folder:
        folder = Path(args.test_folder)
        images = sorted(f for f in folder.iterdir()
                        if f.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            print(f"No images in {folder}")
            sys.exit(1)

        recognizer = ImageRecognizer(args.db_dir)
        total = 0
        for p in images:
            r = recognizer.recognize(str(p))
            total += r.inference_time_ms
            print(f"\n{'─' * 50}")
            print(f"File: {p.name}")
            print(r.format_output())
        print(f"\n{'═' * 50}")
        print(f"Images: {len(images)} | Avg: {total/len(images):.0f}ms")
        return

    if args.camera:
        recognizer = ImageRecognizer(args.db_dir)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: No camera")
            sys.exit(1)
        print("[Camera] SPACE=capture, Q=quit\n")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Kriti Vision", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                r = recognizer.recognize(frame)
                print(r.format_output() + "\n")
        cap.release()
        cv2.destroyAllWindows()
        return

    if args.server:
        run_server(args.server_socket, args.db_dir)
        return

    if args.camera_auto:
        recognizer = ImageRecognizer(args.db_dir)
        selector = FrameSelector(buffer_size=15)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: No camera")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("[Auto Camera] Continuously selecting sharpest frames")
        print("[Auto Camera] Press Q to quit\n")

        last_recognize_time = 0
        RECOGNIZE_INTERVAL = 3.0  # seconds between recognitions

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            sharpness = selector.add_frame(frame)

            # Show feed with sharpness score
            display = frame.copy()
            color = (0, 255, 0) if sharpness > SHARPNESS_THRESHOLD else (0, 0, 255)
            cv2.putText(display, f"Sharpness: {sharpness:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Kriti Vision", display)

            # Auto-recognize every N seconds using best frame
            now = time.time()
            if now - last_recognize_time >= RECOGNIZE_INTERVAL:
                best_frame, best_score = selector.get_best_frame()
                if best_frame is not None and best_score > SHARPNESS_THRESHOLD:
                    print(f"[Auto] Best frame sharpness: {best_score:.1f}")
                    r = recognizer.recognize(best_frame)
                    if r.detections:
                        print(r.format_output() + "\n")
                    selector.clear()
                    last_recognize_time = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        return

    parser.print_help()


if __name__ == "__main__":
    main()