# -*- coding: utf-8 -*-
"""
Process vehicle registration certificates using LOCAL OCR only (no Google Vision).
- Uses PaddleOCR for Korean/English text recognition
- Handles all PDF pages (not just first page)
- Copies processed files to output directory with 차량번호_연료유형 naming
"""
import os
import sys
import time
import gc
import shutil
import argparse
import unicodedata
import logging
from datetime import datetime

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ocr.local_engine import LocalOCREngine
from src.ocr.preprocessor import ImagePreprocessor
from src.parser.car_registration import CarRegistrationParser
from src.validator.vin_validator import VINValidator
from src.storage.gsheets import GoogleSheetClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
GC_INTERVAL = 10
RATE_LIMIT_SECONDS = 0.5  # Faster since no cloud API


def normalize_path(path):
    """Normalize Unicode path for consistent handling."""
    return unicodedata.normalize('NFC', path)


def sanitize_filename(text):
    """Remove invalid characters from filename."""
    if not text:
        return ''
    # Remove/replace invalid filename characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        text = text.replace(char, '_')
    return text.strip()


def copy_to_output(source_path, output_dir, vehicle_no, fuel_type):
    """
    Copy file to output directory with 차량번호_연료유형 naming.

    Args:
        source_path: Original file path
        output_dir: Output directory
        vehicle_no: Vehicle number (e.g., 충남70자1206)
        fuel_type: Fuel type (e.g., Electric, CNG, Diesel)

    Returns:
        str: Path to copied file, or None if copy failed
    """
    if not vehicle_no:
        return None

    # Get original extension
    _, ext = os.path.splitext(source_path)

    # Sanitize components
    vehicle_no = sanitize_filename(vehicle_no)
    fuel_type = sanitize_filename(fuel_type) if fuel_type else 'Unknown'

    # Create filename: 차량번호_연료유형.ext
    new_filename = f"{vehicle_no}_{fuel_type}{ext}"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Handle duplicate filenames
    dest_path = os.path.join(output_dir, new_filename)
    if os.path.exists(dest_path):
        # Add timestamp suffix for duplicates
        timestamp = datetime.now().strftime('%H%M%S')
        new_filename = f"{vehicle_no}_{fuel_type}_{timestamp}{ext}"
        dest_path = os.path.join(output_dir, new_filename)

    try:
        shutil.copy2(source_path, dest_path)
        return dest_path
    except Exception as e:
        logger.error(f"Failed to copy file: {e}")
        return None


def process_files_local(file_list_path, output_dir, start_idx=0, max_files=None,
                        skip_sheet_clear=False, dpi=150):
    """
    Process files using local OCR only.

    Args:
        file_list_path: Path to text file containing one file path per line
        output_dir: Directory to copy processed files (차량번호_연료유형.pdf)
        start_idx: Index to start from (for resuming)
        max_files: Maximum number of files to process (None = all)
        skip_sheet_clear: If True, don't clear the sheet before processing
        dpi: DPI for PDF conversion
    """
    # Read file list
    with open(file_list_path, 'r', encoding='utf-8') as f:
        all_files = [normalize_path(line.strip()) for line in f if line.strip()]

    # Apply start index and max files
    files_to_process = all_files[start_idx:]
    if max_files:
        files_to_process = files_to_process[:max_files]

    total_files = len(files_to_process)
    logger.info(f"Processing {total_files} files (starting from index {start_idx})")
    logger.info(f"Using LOCAL OCR (PaddleOCR) - No cloud API costs")
    logger.info(f"Output directory: {output_dir}")

    # Initialize components
    logger.info("Initializing Local OCR components...")
    engine = LocalOCREngine(lang='korean')
    preprocessor = ImagePreprocessor(dpi=dpi)
    parser = CarRegistrationParser()
    vin_validator = VINValidator()

    logger.info("Initializing Google Sheet...")
    sheet_client = GoogleSheetClient()

    if sheet_client.service and not skip_sheet_clear:
        logger.info("Clearing existing sheet content...")
        sheet_client.clear_first_sheet()
        # Add header
        header = [
            "파일명",           # Filename
            "차량번호",         # Vehicle No
            "차량번호 신뢰도",  # Vehicle No Confidence
            "차대번호(VIN)",    # VIN
            "VIN 신뢰도",       # VIN Confidence
            "VIN 유효성",       # VIN Valid
            "연료타입",         # Fuel Type
            "차종/차명",        # Vehicle Type/Model
            "형식/연식",        # Format/Year
            "원동기형식",       # Engine Type
            "소유자",           # Owner
            "주소",             # Address
            "최초등록일",       # Registration Date
            "제원관리번호",     # Specs Number
            "페이지",           # Page number
            "복사파일",         # Copied file path
            "처리시각"          # Process Time
        ]
        sheet_client.append_row(header)

    logger.info("Starting processing...")
    processed_count = 0
    copied_count = 0
    error_count = 0

    for i, filepath in enumerate(files_to_process):
        actual_idx = start_idx + i
        filename = os.path.basename(filepath)

        logger.info(f"[{actual_idx + 1}/{start_idx + total_files}] Processing: {filename}")

        try:
            # Check if file exists
            if not os.path.exists(filepath):
                logger.warning(f"  File not found, skipping")
                continue

            # Load image - NO max_pages limit for PDFs
            images = preprocessor.load_image(filepath, max_pages=None)
            if not images:
                logger.warning(f"  No images extracted, skipping")
                continue

            logger.info(f"  Found {len(images)} page(s)")

            # Track best result for file copying (use first valid result)
            best_vehicle_no = None
            best_fuel_type = None
            file_copied = False

            for page_idx, image_path in enumerate(images):
                page_num = page_idx + 1
                logger.info(f"  Processing page {page_num}/{len(images)}...")

                # Run Local OCR
                ocr_result = engine.detect_text_hybrid(image_path)

                text = ocr_result.get('google', {}).get('text', '')
                if not text:
                    logger.warning(f"    No text detected on page {page_num}")
                    continue

                # Verify document type
                if not parser.verify_document_type(text):
                    logger.info(f"    Page {page_num} is not a vehicle registration certificate")
                    continue

                # Parse data
                parsed_data = parser.parse_hybrid(ocr_result, filename=filename)
                confidences = parsed_data.get('confidences', {})

                # Validate VIN
                vin = parsed_data.get('vin')
                vehicle_no = parsed_data.get('vehicle_no')
                is_valid, validation_msg = vin_validator.validate(vin)

                # Confidence values
                conf_veh = confidences.get('vehicle_no', 0.0)
                conf_vin = confidences.get('vin', 0.0)

                # Combined fields
                vehicle_type = parsed_data.get('vehicle_type', '')
                model_name = parsed_data.get('model_name', '')
                combined_name = f"{vehicle_type} / {model_name}" if vehicle_type else model_name

                vehicle_format = parsed_data.get('vehicle_format', '')
                model_year = parsed_data.get('model_year', '')
                combined_year = f"{vehicle_format} / {model_year}" if vehicle_format else model_year

                fuel_type = parsed_data.get('fuel_type', 'Unknown')

                # Track best result for copying
                if vehicle_no and not best_vehicle_no:
                    best_vehicle_no = vehicle_no
                    best_fuel_type = fuel_type

                # Copy file (only once per source file)
                copied_path = ""
                if vehicle_no and not file_copied:
                    copied_path = copy_to_output(filepath, output_dir, vehicle_no, fuel_type)
                    if copied_path:
                        file_copied = True
                        copied_count += 1
                        logger.info(f"    Copied to: {os.path.basename(copied_path)}")

                # Build row
                row_data = [
                    filename,                                    # 파일명
                    vehicle_no or '',                           # 차량번호
                    f"{conf_veh:.4f}",                          # 차량번호 신뢰도
                    vin or '',                                  # 차대번호
                    f"{conf_vin:.4f}",                          # VIN 신뢰도
                    "Valid" if is_valid else validation_msg,    # VIN 유효성
                    fuel_type,                                  # 연료타입
                    combined_name,                              # 차종/차명
                    combined_year,                              # 형식/연식
                    parsed_data.get('engine_type', ''),         # 원동기형식
                    parsed_data.get('owner_name', ''),          # 소유자
                    parsed_data.get('owner_address', ''),       # 주소
                    parsed_data.get('registration_date', ''),   # 최초등록일
                    parsed_data.get('vehicle_specs', ''),       # 제원관리번호
                    str(page_num),                              # 페이지
                    os.path.basename(copied_path) if copied_path else '',  # 복사파일
                    time.strftime("%Y-%m-%d %H:%M:%S")          # 처리시각
                ]

                # Upload to sheet
                if sheet_client.service:
                    sheet_client.append_row(row_data)

                logger.info(f"    -> VehNo: {vehicle_no}, VIN: {vin}, Fuel: {fuel_type}")

                # Cleanup temp files for this page
                if image_path != filepath:
                    ImagePreprocessor.cleanup_temp_files([image_path], filepath)

            processed_count += 1

        except Exception as e:
            logger.error(f"  Error: {e}")
            error_count += 1

        # Garbage collection
        if (i + 1) % GC_INTERVAL == 0:
            gc.collect()
            logger.debug(f"  [GC] Collected after {i + 1} files")

        # Rate limiting (reduced since no cloud API)
        time.sleep(RATE_LIMIT_SECONDS)

    # Final cleanup
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing complete!")
    logger.info(f"  Files processed: {processed_count}")
    logger.info(f"  Files copied: {copied_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Last index: {start_idx + total_files - 1}")
    logger.info(f"{'='*50}")

    LocalOCREngine.cleanup()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description='Process vehicle registration files using LOCAL OCR (no cloud API)'
    )
    parser.add_argument('--file-list',
                        default='/Users/ssh/Documents/Develope/OCR_Multi_02/dropbox_registration_files.txt',
                        help='Path to file containing list of files to process')
    parser.add_argument('--output-dir',
                        default='/Users/ssh/Documents/Develope/OCR_Multi_02/output',
                        help='Directory to copy processed files (차량번호_연료유형.pdf)')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting index for resuming (default: 0)')
    parser.add_argument('--max', type=int, default=None,
                        help='Maximum number of files to process (default: all)')
    parser.add_argument('--skip-clear', action='store_true',
                        help='Skip clearing the sheet (for resuming)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for PDF conversion (default: 150)')

    args = parser.parse_args()

    process_files_local(
        file_list_path=args.file_list,
        output_dir=args.output_dir,
        start_idx=args.start,
        max_files=args.max,
        skip_sheet_clear=args.skip_clear,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
