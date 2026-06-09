#!/bin/bash
# ──────────────────────────────────────────────────────────
# Kriti 2026 — RPi Vision Setup & Start Script
# ──────────────────────────────────────────────────────────
# Run this once on a fresh Raspberry Pi:
#   chmod +x start_rpi.sh
#   ./start_rpi.sh setup
#
# Then to start the camera pipeline:
#   ./start_rpi.sh run
#
# Or just run recognition on an image:
#   ./start_rpi.sh test image.jpg
# ──────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
VENV_DIR="$SCRIPT_DIR/venv"

# ── Colours ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ──────────────────────────────────────────────
# SETUP — install everything from scratch
# ──────────────────────────────────────────────
do_setup() {
    info "Starting full RPi setup..."

    # System packages
    info "Installing system packages..."
    sudo apt update
    sudo apt install -y \
        python3-pip python3-venv git \
        libzbar0t64 tesseract-ocr \
        cmake libopenblas-dev liblapack-dev \
        libhdf5-dev \
        libcamera-apps python3-picamera2

    # Create venv
    if [ ! -d "$VENV_DIR" ]; then
        info "Creating Python virtual environment..."
        python3 -m venv --system-site-packages venv
    fi
    source "$VENV_DIR/bin/activate"

    # Python packages
    info "Installing Python packages (this will take a while)..."
    pip install --upgrade pip

    info "Installing core packages..."
    pip install numpy opencv-python-headless Pillow pyzbar pytesseract

    info "Installing PyTorch (CPU)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

    info "Installing ultralytics (YOLO)..."
    pip install ultralytics

    info "Installing face_recognition (dlib build takes ~20min on Pi)..."
    pip install dlib face_recognition

    info ""
    info "========================================="
    info "  SETUP COMPLETE"
    info "========================================="
    info "Run the camera:  ./start_rpi.sh run"
    info "Test an image:   ./start_rpi.sh test image.jpg"
    info "Camera test:     ./start_rpi.sh camtest"
}

# ──────────────────────────────────────────────
# ACTIVATE venv helper
# ──────────────────────────────────────────────
activate_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        error "Virtual environment not found. Run: ./start_rpi.sh setup"
        exit 1
    fi
    source "$VENV_DIR/bin/activate"
}

# ──────────────────────────────────────────────
# RUN — start the camera pipeline
# ──────────────────────────────────────────────
do_run() {
    activate_venv
    info "Starting camera pipeline (auto mode, headless)..."
    info "Press Ctrl+C to stop"
    echo ""
    python3 camera_pipeline.py --auto --headless --db-dir ./db
}

# ──────────────────────────────────────────────
# STREAM — camera pipeline + live browser feed
# ──────────────────────────────────────────────
do_stream() {
    activate_venv
    info "Starting camera pipeline with live web stream..."
    info "Open browser on laptop: http://$(hostname -I | awk '{print $1}'):8080"
    info "Press Ctrl+C to stop"
    echo ""
    python3 camera_pipeline.py --auto --stream --headless --db-dir ./db
}

# ──────────────────────────────────────────────
# TEST — run recognition on a single image
# ──────────────────────────────────────────────
do_test() {
    activate_venv
    if [ -z "$1" ]; then
        error "Usage: ./start_rpi.sh test <image_path>"
        exit 1
    fi
    python3 recognize_mobilenet.py --image "$1"
}

# ──────────────────────────────────────────────
# CAMTEST — quick camera check (no recognition)
# ──────────────────────────────────────────────
do_camtest() {
    activate_venv
    info "Testing camera (saving test_frame.jpg)..."
    python3 -c "
from picamera2 import Picamera2
import cv2, time
cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={'size': (1280,720), 'format': 'BGR888'}))
cam.start()
time.sleep(1)
frame = cam.capture_array()
cv2.imwrite('test_frame.jpg', frame)
cam.stop()
print(f'Saved test_frame.jpg ({frame.shape[1]}x{frame.shape[0]})')
"
    info "Done. Check test_frame.jpg"
}

# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
case "${1:-}" in
    setup)
        do_setup
        ;;
    run)
        do_run
        ;;
    stream)
        do_stream
        ;;
    test)
        do_test "$2"
        ;;
    camtest)
        do_camtest
        ;;
    *)
        echo "Kriti 2026 — RPi Vision"
        echo ""
        echo "Usage: ./start_rpi.sh <command>"
        echo ""
        echo "Commands:"
        echo "  setup    Install all dependencies (run once on fresh Pi)"
        echo "  run      Start live camera pipeline (auto A4 trigger)"
        echo "  stream   Same as run + live video feed in browser"
        echo "  test     Run recognition on a single image"
        echo "  camtest  Quick camera test (saves test_frame.jpg)"
        echo ""
        echo "Examples:"
        echo "  ./start_rpi.sh setup"
        echo "  ./start_rpi.sh run"
        echo "  ./start_rpi.sh stream"
        echo "  ./start_rpi.sh test roger.jpeg"
        echo "  ./start_rpi.sh camtest"
        ;;
esac
