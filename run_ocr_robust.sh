#!/bin/bash
# OCR Processing Script with Memory Optimization
# Usage: ./run_ocr_robust.sh [options]
#
# Options:
#   --no-paddle     Disable PaddleOCR (Google Vision only, more stable)
#   --dpi N         Set PDF conversion DPI (default: 150, lower = less memory)
#   --batch-size N  Files per batch (default: 50)
#   --input_dir DIR Input directory (default: ./input)

set -e

# Activate virtual environment
source .venv/bin/activate
export PYTHONPATH=.

# Default: Run with optimized settings (no PaddleOCR for stability)
python -m src.main --no-paddle --dpi 150 "$@"
