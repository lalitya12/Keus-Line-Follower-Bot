#!/usr/bin/env python3
"""
Kriti 2026 - Raspberry Pi Camera Pipeline
==========================================
Dedicated camera capture module for Raspberry Pi.

HOW IT WORKS:
  1. Continuously reads frames from Pi camera
  2. Draws a live A4 guide overlay on screen
  3. Detects A4 sheet using temporal stability (must be seen N frames in a row)
  4. Waits for camera to settle (sharpness check) then picks best frame
  5. Runs the full recognition pipeline in a background thread
  6. Displays result on screen + prints to terminal

USAGE:
  python camera_pipeline.py --auto      # Auto-trigger when A4 is stable
  python camera_pipeline.py --manual    # Press SPACE to capture
  python camera_pipeline.py --test      # Just test A4 detection, no recognition
  python camera_pipeline.py --auto --db-dir ./db

INSTALL (Raspberry Pi):
  sudo apt-get install -y libzbar0
  pip install opencv-python-headless numpy --break-system-packages
  pip install picamera2 --break-system-packages   # Pi Camera Module
"""

import time
import sys
import argparse
import threading
import queue
from collections import deque
from pathlib import Path
import subprocess
import io
import socketio
import base64

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

FRAME_WIDTH        = 1280
FRAME_HEIGHT       = 720
SHARPNESS_THRESH   = 50.0    # Laplacian variance — below = too blurry to capture
BUFFER_SIZE        = 10      # rolling frame buffer size

# A4 detection — moving camera robustness
MIN_A4_AREA_FRAC   = 0.08    # sheet must cover at least 8% of frame
MAX_A4_AREA_FRAC   = 0.93    # sheet must not fill more than 93% of frame
A4_STABILITY_FRAMES = 2      # must detect A4 for this many consecutive frames
A4_IOU_THRESHOLD   = 0.50    # min IoU between consecutive detections to count stable
BRIGHTNESS_MIN     = 130     # inner region mean brightness for paper white check

COOLDOWN_SEC       = 5.0     # min seconds between auto-triggers
RESULT_DISPLAY_SEC = 4.0     # seconds to show result overlay

# ── Colors (BGR) — natural, no blue/green tint ──
WHITE      = (255, 255, 255)
BLACK      = (0,   0,   0)
GRAY_DARK  = (40,  40,  40)
GRAY_MID   = (110, 110, 110)
ORANGE     = (30,  140, 255)   # guide / waiting — warm orange
GREEN_OK   = (60,  200,  60)   # confirmed / stable — clean green
RED_WARN   = (55,   55, 210)   # warning / not sharp
YELLOW_HUD = (30,  210, 230)   # HUD accent — warm yellow
PANEL_BG   = (22,  22,  22)    # result panel background


# ─────────────────────────────────────────────────────────
# CAMERA BACKEND
# ─────────────────────────────────────────────────────────

class CameraBackend:
    """
    Unified camera interface.
    Uses rpicam-vid via subprocess to capture an MJPEG stream, 
    preserving native hardware color grading and AWB.
    """

    def __init__(self, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self.process = None
        self.width  = width
        self.height = height
        self.byte_data = b''
        self._init_camera()

    def _init_camera(self):
        cmd = [
            'rpicam-vid',
            '-t', '0',
            '--width', str(self.width),
            '--height', str(self.height),
            '--framerate', '15',
            '--inline',            
            '--nopreview',         
            '--codec', 'mjpeg',
            '-o', '-'
        ]
        
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            print(f"[Camera] rpicam-vid started ({self.width}x{self.height})")
        except Exception as e:
            raise RuntimeError(f"Failed to start rpicam-vid: {e}")

    def read(self):
        """Returns (success: bool, frame: np.ndarray BGR)."""
        if not self.process:
            return False, None

        while True:
            chunk = self.process.stdout.read(8192)
            if not chunk:
                return False, None
                
            self.byte_data += chunk
            
            a = self.byte_data.find(b'\xff\xd8')
            b = self.byte_data.find(b'\xff\xd9')
            
            if a != -1 and b != -1:
                if a < b:
                    jpg = self.byte_data[a:b+2]
                    self.byte_data = self.byte_data[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        return True, frame
                    else:
                        return False, None
                else:
                    self.byte_data = self.byte_data[a:]

    def release(self):
        if self.process:
            self.process.terminate()
            self.process.wait()


# ─────────────────────────────────────────────────────────
# SHARPNESS
# ─────────────────────────────────────────────────────────

def compute_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ─────────────────────────────────────────────────────────
# FRAME BUFFER
# ─────────────────────────────────────────────────────────

class FrameBuffer:
    def __init__(self, size=BUFFER_SIZE):
        self.size   = size
        self.frames = deque()

    def push(self, frame: np.ndarray) -> float:
        score = compute_sharpness(frame)
        self.frames.append((frame.copy(), score))
        if len(self.frames) > self.size:
            self.frames.popleft()
        return score

    def best(self):
        if not self.frames:
            return None, 0.0
        return max(self.frames, key=lambda x: x[1])

    def clear(self):
        self.frames.clear()


# ─────────────────────────────────────────────────────────
# A4 SHEET DETECTION
# Finds the A4 paper in the frame using edge detection,
# contour finding, and polygon approximation.
# Returns 4-corner polygon + bounding rect.
# ─────────────────────────────────────────────────────────

A4_ASPECT_LO = 1.1    # A4 ≈ 1.414; allow some skew
A4_ASPECT_HI = 2.0
A4_POLY_SIDES = 4     # we want a quadrilateral


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 corners: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    rect[1] = pts[np.argmin(d)]   # top-right
    rect[3] = pts[np.argmax(d)]   # bottom-left
    return rect


def find_a4_contour(frame: np.ndarray):
    """
    Detect A4 sheet (white rectangular paper) in the frame.
    
    Returns:
        (corners_4x2, (x, y, w, h))  — 4 ordered corner points + bounding rect
        or None if no A4 found.
    """
    fh, fw = frame.shape[:2]
    frame_area = fh * fw

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold to separate white paper from background
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Try multiple edge approaches and pick best result
    candidates = []

    # --- Method 1: Canny edges ---
    edges1 = cv2.Canny(blurred, 40, 120)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges1 = cv2.dilate(edges1, k, iterations=1)
    candidates.append(edges1)

    # --- Method 2: Adaptive threshold ---
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    thresh_inv = cv2.bitwise_not(thresh)
    thresh_inv = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                                   iterations=2)
    candidates.append(thresh_inv)

    # --- Method 3: Simple binary threshold for white paper ---
    _, white_mask = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
                                   iterations=2)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                                   iterations=1)
    candidates.append(white_mask)

    best_score = 0
    best_result = None

    for edge_img in candidates:
        contours, _ = cv2.findContours(
            edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            area = cv2.contourArea(cnt)

            # Area check
            if area < frame_area * MIN_A4_AREA_FRAC:
                continue
            if area > frame_area * MAX_A4_AREA_FRAC:
                continue

            # Approximate polygon
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) != A4_POLY_SIDES:
                continue

            # Must be convex
            if not cv2.isContourConvex(approx):
                continue

            # Aspect ratio check on bounding rect
            x, y, w, h = cv2.boundingRect(approx)
            if w == 0 or h == 0:
                continue
            aspect = max(w, h) / min(w, h)
            if aspect < A4_ASPECT_LO or aspect > A4_ASPECT_HI:
                continue

            # Brightness check — inner region should be white-ish (paper)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [approx], -1, 255, -1)
            # Erode mask slightly to avoid border pixels
            eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
            if np.count_nonzero(eroded) > 100:
                mean_brightness = cv2.mean(gray, mask=eroded)[0]
                if mean_brightness < BRIGHTNESS_MIN:
                    continue

            # Score: prefer larger area contours with good rectangularity
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            score = area * rectangularity

            if score > best_score:
                best_score = score
                corners = approx.reshape(4, 2).astype(np.float32)
                ordered = _order_corners(corners)
                best_result = (ordered, (x, y, w, h))

    return best_result


# ─────────────────────────────────────────────────────────
# TEMPORAL STABILITY
# ─────────────────────────────────────────────────────────

def _iou_xywh(a, b) -> float:
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = ax1 + a[2], ay1 + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = bx1 + b[2], by1 + b[3]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


class A4Tracker:
    """
    Tracks the A4 sheet detection across frames.
    stable=True when the A4 bounding box has been consistent
    for A4_STABILITY_FRAMES consecutive frames.
    """

    def __init__(self, required=A4_STABILITY_FRAMES, iou_thresh=A4_IOU_THRESHOLD):
        self.required     = required
        self.iou_thresh   = iou_thresh
        self._streak      = 0
        self._last_roi    = None
        self.stable       = False
        self.corners      = None   # 4x2 float32 ordered corner points
        self.roi          = None   # (x, y, w, h) bounding rect

    @property
    def streak(self):
        return self._streak

    def update(self, found: bool, detection=None):
        """
        found: bool — whether an A4 sheet was detected
        detection: (corners_4x2, (x,y,w,h)) from find_a4_contour, or None
        """
        if not found or detection is None:
            self._streak   = 0
            self._last_roi = None
            self.stable    = False
            self.corners   = None
            self.roi       = None
            return

        corners, roi = detection

        if self._last_roi is not None:
            iou = _iou_xywh(roi, self._last_roi)
            if iou >= self.iou_thresh:
                self._streak += 1
            else:
                self._streak = 1
        else:
            self._streak = 1

        self._last_roi = roi
        self.corners   = corners
        self.roi       = roi
        self.stable    = (self._streak >= self.required)

    def reset(self):
        self._streak   = 0
        self._last_roi = None
        self.stable    = False
        self.corners   = None
        self.roi       = None


# ─────────────────────────────────────────────────────────
# DISPLAY HELPERS  — clean, natural colors
# ─────────────────────────────────────────────────────────

_FONT = cv2.FONT_HERSHEY_DUPLEX


def _text(img, txt, pos, scale=0.6, color=WHITE, thick=1, shadow=True):
    x, y = pos
    if shadow:
        cv2.putText(img, txt, (x + 1, y + 1), _FONT, scale,
                    BLACK, thick + 1, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), _FONT, scale, color, thick, cv2.LINE_AA)


def draw_guide(frame: np.ndarray, tracker: A4Tracker,
               raw_found: bool, raw_detection=None,
               sharpness: float = 0.0) -> np.ndarray:
    display = frame.copy()
    h, w    = display.shape[:2]

    # ── A4 polygon outline ──
    if tracker.stable and tracker.corners is not None:
        # Draw tight green polygon around detected A4 corners
        pts = tracker.corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(display, [pts], True, GREEN_OK, 3, cv2.LINE_AA)
        # Corner L-marks at each detected corner
        arm = 20
        for i in range(4):
            cx, cy = int(tracker.corners[i][0]), int(tracker.corners[i][1])
            # Draw small cross at each corner
            cv2.circle(display, (cx, cy), 5, GREEN_OK, -1, cv2.LINE_AA)

    elif raw_found and raw_detection is not None:
        corners, (rx, ry, rw, rh) = raw_detection
        # Draw orange polygon while stabilizing
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(display, [pts], True, ORANGE, 2, cv2.LINE_AA)
        # Small dots at corners
        for i in range(4):
            cx, cy = int(corners[i][0]), int(corners[i][1])
            cv2.circle(display, (cx, cy), 4, ORANGE, -1, cv2.LINE_AA)

    else:
        # Corner guide markers only — hint where to place the paper
        mx, my = int(w * 0.10), int(h * 0.10)
        corners_guide = [(mx, my), (w - mx, my), (w - mx, h - my), (mx, h - my)]
        arm = 36
        offsets = [( arm, 0, 0,  arm),
                   (-arm, 0, 0,  arm),
                   (-arm, 0, 0, -arm),
                   ( arm, 0, 0, -arm)]
        for (gx, gy), (dx, _, _, dy) in zip(corners_guide, offsets):
            cv2.line(display, (gx, gy), (gx + dx, gy), GRAY_MID, 2, cv2.LINE_AA)
            cv2.line(display, (gx, gy), (gx, gy + dy), GRAY_MID, 2, cv2.LINE_AA)

    # ── Status text ──
    streak = tracker.streak
    if tracker.stable:
        msg = "A4 locked — capturing..."
        col = GREEN_OK
    elif raw_found:
        msg = f"A4 detected — hold steady  {streak}/{A4_STABILITY_FRAMES}"
        col = ORANGE
    else:
        msg = "Point camera at A4 sheet"
        col = WHITE
    _text(display, msg, (18, 38), scale=0.72, color=col)

    # ── Sharpness bar ──
    bar_w = 160; bar_h = 10
    bx = w - bar_w - 16; by = h - 28
    fill   = min(int(bar_w * min(sharpness, 300) / 300), bar_w)
    barcol = GREEN_OK if sharpness >= SHARPNESS_THRESH else RED_WARN
    cv2.rectangle(display, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
    cv2.rectangle(display, (bx, by), (bx + fill, by + bar_h), barcol, -1)
    _text(display, f"Sharp {sharpness:.0f}",
          (bx, by - 8), scale=0.38, color=GRAY_MID, shadow=False)

    return display


def draw_result_overlay(frame: np.ndarray, result_text: str,
                        cropped: np.ndarray | None) -> np.ndarray:
    """Semi-transparent right panel with result. Dark neutral background."""
    display = frame.copy()
    h, w    = display.shape[:2]
    pw      = 320

    panel_region = display[:, w - pw:].copy()
    blended      = cv2.addWeighted(panel_region, 0.15,
                                   np.full_like(panel_region, 22), 0.85, 0)
    display[:, w - pw:] = blended
    cv2.line(display, (w - pw, 0), (w - pw, h), (75, 75, 75), 1)

    y = 16
    if cropped is not None:
        th = 148; tw = int(th * cropped.shape[1] / cropped.shape[0])
        tw = min(tw, pw - 20)
        thumb = cv2.resize(cropped, (tw, th))
        tx    = w - pw + (pw - tw) // 2
        display[y: y + th, tx: tx + tw] = thumb
        cv2.rectangle(display, (tx, y), (tx + tw, y + th), (85, 85, 85), 1)
        y += th + 14

    _text(display, "RESULT", (w - pw + 12, y),
          scale=0.55, color=YELLOW_HUD, thick=1)
    cv2.line(display, (w - pw + 10, y + 7), (w - 10, y + 7), (65, 65, 65), 1)
    y += 22

    for line in result_text.split("\n"):
        s = line.strip()
        if not s:
            y += 5; continue
        if s.startswith("Category:") or s.startswith("Content:"):
            col = WHITE
        elif s.startswith("["):
            col = GRAY_MID
        else:
            col = (185, 185, 185)
        _text(display, s[:38], (w - pw + 12, y),
              scale=0.42, color=col, shadow=False)
        y += 20
        if y > h - 10:
            break

    return display

# ─────────────────────────────────────────────────────────
# DASHBOARD SOCKET CONNECTOR
# ─────────────────────────────────────────────────────────

sio = socketio.Client()

def connect_dashboard(port: int = 8080):
    """Attempt to connect to the Flask app.py server."""
    try:
        sio.connect(f'http://127.0.0.1:{port}')
        print(f"[SocketIO] Connected to dashboard on port {port}")
    except Exception as e:
        print(f"[SocketIO] Warning: Could not connect to dashboard - {e}")

def emit_video_frame(display: np.ndarray):
    """Encode the live frame to base64 and fire it to the dashboard."""
    if not sio.connected:
        return
    
    # Compress slightly to ensure smooth websocket transmission
    ok, jpg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if ok:
        b64_img = base64.b64encode(jpg.tobytes()).decode('utf-8')
        try:
            sio.emit('video_frame', {'image': b64_img})
        except Exception:
            pass

def emit_recognition(result_text: str, cropped_img: np.ndarray | None):
    """Fire the recognized text and cropped target image to the dashboard."""
    if not sio.connected:
        return
        
    b64_crop = ""
    if cropped_img is not None:
        # High quality for the saved target image
        ok, c_jpg = cv2.imencode(".jpg", cropped_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            b64_crop = base64.b64encode(c_jpg.tobytes()).decode('utf-8')
            
    try:
        sio.emit('recognition_result', {
            'text': result_text, 
            'image': b64_crop
        })
    except Exception:
        pass


# ─────────────────────────────────────────────────────────
# RECOGNITION WORKER  (background thread)
# ─────────────────────────────────────────────────────────

def recognition_worker(task_q: queue.Queue, result_q: queue.Queue, db_dir: str):
    print("[Worker] Loading models…")
    sys.path.insert(0, str(Path(__file__).parent))
    recognizer = None
    try:
        from recognize_mobilenet import ImageRecognizer
        recognizer = ImageRecognizer(db_dir)
        print("[Worker] Ready.")
    except ImportError as e:
        print(f"[Worker] WARNING: could not load recognition module: {e}")

    while True:
        item = task_q.get()
        if item is None:
            break
        frame, cropped = item
        if recognizer is None:
            result_q.put(("Recognition module not loaded", cropped))
            continue
        try:
            res = recognizer.recognize(frame)
            result_q.put((res.format_output(), cropped))
        except Exception as e:
            result_q.put((f"Error: {e}", cropped))


# ─────────────────────────────────────────────────────────
# CAPTURE HELPER
# ─────────────────────────────────────────────────────────

def _do_capture(buf: FrameBuffer, tracker: A4Tracker, task_q: queue.Queue):
    best_frame, best_sharp = buf.best()
    if best_frame is None:
        return
    print(f"[Capture] Sharpness={best_sharp:.1f}")
    # Crop to A4 bounding rect for thumbnail; pass full frame to recognizer
    crop = None
    if tracker.roi is not None:
        rx, ry, rw, rh = tracker.roi
        crop = best_frame[ry:ry+rh, rx:rx+rw].copy()
    if not task_q.full():
        task_q.put_nowait((best_frame, crop))


# ─────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────

def run_camera(auto_mode: bool, db_dir: str, headless: bool = False,
               stream_port: int = 0):
    cam     = CameraBackend(FRAME_WIDTH, FRAME_HEIGHT)
    buf     = FrameBuffer(BUFFER_SIZE)
    tracker = A4Tracker()

    task_q  = queue.Queue(maxsize=1)
    result_q = queue.Queue()
    worker  = threading.Thread(
        target=recognition_worker, args=(task_q, result_q, db_dir), daemon=True
    )
    worker.start()

    last_trigger = 0.0
    processing   = False
    last_result  = None
    result_until = 0.0
    fps = 0.0; fps_n = 0; fps_t = time.time()

    mode = "AUTO (A4 stable trigger)" if auto_mode else "MANUAL (SPACE to capture)"
    print(f"[Camera] {mode} | Q=quit | S=save frame\n")

    if stream_port:
        connect_dashboard(stream_port)

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue

            sharpness = buf.push(frame)
            fps_n += 1
            if fps_n >= 20:
                fps   = fps_n / max(time.time() - fps_t, 0.001)
                fps_n = 0
                fps_t = time.time()

            # A4 sheet detection
            detection = find_a4_contour(frame)
            found = detection is not None
            tracker.update(found, detection)

            if not result_q.empty():
                last_result  = result_q.get_nowait()
                processing   = False
                result_until = time.time() + RESULT_DISPLAY_SEC

                if stream_port:
                    emit_recognition(last_result[0], last_result[1])
                
                print("\n" + "─" * 54)
                print(last_result[0])
                print("─" * 54 + "\n")

            now = time.time()
            if not processing and now - last_trigger >= COOLDOWN_SEC:
                if auto_mode and tracker.stable and sharpness >= SHARPNESS_THRESH:
                    _do_capture(buf, tracker, task_q)
                    processing   = True
                    last_trigger = now
                    tracker.reset()
                    buf.clear()

            if not headless:
                showing = last_result is not None and now < result_until
                if showing:
                    display = draw_result_overlay(frame, *last_result)
                else:
                    display = draw_guide(frame, tracker, found, detection, sharpness)

                if processing:
                    _text(display, "Processing...", (18, 72),
                          scale=0.65, color=ORANGE)

                _text(display, f"{fps:.0f} fps",
                      (16, frame.shape[0] - 10), scale=0.38,
                      color=GRAY_MID, shadow=False)

                # Push to web stream if enabled
                if stream_port:
                    emit_video_frame(display)

                cv2.imshow("Kriti Vision", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord(" ") and not auto_mode and not processing:
                    _do_capture(buf, tracker, task_q)
                    processing   = True
                    last_trigger = now
                    tracker.reset()
                    buf.clear()
                    print(f"[Manual] Sharp={sharpness:.1f}")
                elif key == ord("s"):
                    fname = f"cap_{int(now)}.jpg"
                    cv2.imwrite(fname, frame)
                    print(f"[Saved] {fname}")
            else:
                # headless — still build display frame for stream if needed
                if stream_port:
                    showing = last_result is not None and now < result_until
                    if showing:
                        display = draw_result_overlay(frame, *last_result)
                    else:
                        display = draw_guide(frame, tracker, found, detection, sharpness)
                    if processing:
                        _text(display, "Processing...", (18, 72),
                              scale=0.65, color=ORANGE)
                    emit_video_frame(display)
                time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n[Camera] Stopped.")
    finally:
        task_q.put(None)
        cam.release()
        if not headless:
            cv2.destroyAllWindows()
        worker.join(timeout=3)


# ─────────────────────────────────────────────────────────
# TEST MODE
# ─────────────────────────────────────────────────────────

def run_test():
    """Test A4 sheet detection — no recognition needed."""
    print("[Test] A4 sheet detection — Q=quit | S=save\n")
    cam     = CameraBackend(FRAME_WIDTH, FRAME_HEIGHT)
    buf     = FrameBuffer(BUFFER_SIZE)
    tracker = A4Tracker()

    while True:
        ok, frame = cam.read()
        if not ok:
            continue
        sharpness = buf.push(frame)
        detection = find_a4_contour(frame)
        found = detection is not None
        tracker.update(found, detection)

        display = draw_guide(frame, tracker, found, detection, sharpness)

        # Mini crop preview top-left
        if found and detection is not None:
            _, (rx, ry, rw, rh) = detection
            crop = frame[ry:ry+rh, rx:rx+rw]
            if crop.size > 0:
                tw = 120; th = int(tw * rh / max(rw, 1))
                th = min(th, 120)
                thumb = cv2.resize(crop, (tw, th))
                display[8: 8+th, 8: 8+tw] = thumb
                cv2.rectangle(display, (8, 8), (8+tw, 8+th), (80, 80, 80), 1)

        if tracker.stable:
            _text(display, "A4 STABLE — would trigger",
                  (18, display.shape[0] - 45),
                  scale=0.6, color=GREEN_OK)

        cv2.imshow("Focus Test", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(f"test_{int(time.time())}.jpg", frame)
            print("[Saved]")

    cam.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Kriti 2026 — Pi Camera Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python camera_pipeline.py --test                          # verify camera + focus detection
  python camera_pipeline.py --auto                          # auto-trigger when focus stable
  python camera_pipeline.py --manual                        # SPACE to capture
  python camera_pipeline.py --auto --headless               # SSH / no display
  python camera_pipeline.py --stream                        # web stream only on :8080
  python camera_pipeline.py --auto --stream                 # auto + web stream
  python camera_pipeline.py --auto --stream --headless      # headless + web stream
  python camera_pipeline.py --auto --stream --stream-port 9090
  python camera_pipeline.py --auto --db-dir ./db
        """
    )
    p.add_argument("--auto",     action="store_true")
    p.add_argument("--manual",   action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--test",     action="store_true")
    p.add_argument("--stream",   action="store_true",
                   help="Start MJPEG web stream on --stream-port (default 8080)")
    p.add_argument("--stream-port", type=int, default=8080,
                   help="Port for the MJPEG web stream (default: 8080)")
    p.add_argument("--db-dir",   default="./db")
    p.add_argument("--width",    type=int, default=FRAME_WIDTH)
    p.add_argument("--height",   type=int, default=FRAME_HEIGHT)
    args = p.parse_args()

    if args.test:
        run_test()
    elif args.auto or args.manual or args.stream:
        run_camera(
            auto_mode   = args.auto,
            db_dir      = args.db_dir,
            headless    = args.headless,
            stream_port = args.stream_port if args.stream else 0,
        )
    else:
        p.print_help()


if __name__ == "__main__":
    main()
