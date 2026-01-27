# -*- coding: utf-8 -*-
"""
Process vehicle registration certificates from Dropbox directory.
This script reads files from a list and processes them without moving originals.
"""
import os
import sys
import time
import gc
import argparse
import unicodedata
from dotenv import load_dotenv

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ocr.hybrid_engine import HybridOCREngine
from src.ocr.preprocessor import ImagePreprocessor
from src.parser.car_registration import CarRegistrationParser
from src.validator.vin_validator import VINValidator
from src.storage.gsheets import GoogleSheetClient

# Constants
GC_INTERVAL = 10
RATE_LIMIT_SECONDS = 1.5


def normalize_path(path):
    """Normalize Unicode path for consistent handling."""
    return unicodedata.normalize('NFC', path)


def process_dropbox_files(file_list_path, start_idx=0, max_files=None, skip_sheet_clear=False):
    """
    Process files from a list file.

    Args:
        file_list_path: Path to text file containing one file path per line
        start_idx: Index to start from (for resuming)
        max_files: Maximum number of files to process (None = all)
        skip_sheet_clear: If True, don't clear the sheet before processing
    """
    load_dotenv()
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') is None:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service_account.json'

    # Read file list
    with open(file_list_path, 'r', encoding='utf-8') as f:
        all_files = [normalize_path(line.strip()) for line in f if line.strip()]

    # Apply start index and max files
    files_to_process = all_files[start_idx:]
    if max_files:
        files_to_process = files_to_process[:max_files]

    total_files = len(files_to_process)
    print(f"Processing {total_files} files (starting from index {start_idx})")

    # Initialize components
    print("Initializing OCR components...")
    engine = HybridOCREngine(enable_paddle=False)  # Google Vision only for stability
    preprocessor = ImagePreprocessor(dpi=150)
    parser = CarRegistrationParser()
    vin_validator = VINValidator()

    print("Initializing Google Sheet...")
    sheet_client = GoogleSheetClient()

    if sheet_client.service and not skip_sheet_clear:
        print("Clearing existing sheet content...")
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
            "처리시각"          # Process Time
        ]
        sheet_client.append_row(header)

    print("Starting processing...")
    processed_count = 0
    error_count = 0

    for i, filepath in enumerate(files_to_process):
        actual_idx = start_idx + i
        filename = os.path.basename(filepath)

        print(f"[{actual_idx + 1}/{start_idx + total_files}] Processing: {filename}")

        try:
            # Check if file exists
            if not os.path.exists(filepath):
                print(f"  File not found, skipping")
                continue

            # Load image (first page only for PDFs)
            images = preprocessor.load_image(filepath, max_pages=1)
            if not images:
                print(f"  No images extracted, skipping")
                continue

            for page_idx, image_path in enumerate(images):
                # Run Hybrid OCR
                hybrid_result = engine.detect_text_hybrid(image_path)

                g_text = hybrid_result.get('google', {}).get('text', '')
                if not g_text:
                    print(f"  No text detected on page {page_idx + 1}")
                    continue

                # Verify document type
                if not parser.verify_document_type(g_text):
                    print(f"  Not a vehicle registration certificate")
                    continue

                # Parse data
                parsed_data = parser.parse_hybrid(hybrid_result, filename=filename)
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

                # Build row
                row_data = [
                    filename,                                    # 파일명
                    vehicle_no or '',                           # 차량번호
                    f"{conf_veh:.4f}",                          # 차량번호 신뢰도
                    vin or '',                                  # 차대번호
                    f"{conf_vin:.4f}",                          # VIN 신뢰도
                    "Valid" if is_valid else validation_msg,    # VIN 유효성
                    parsed_data.get('fuel_type', 'Unknown'),    # 연료타입
                    combined_name,                              # 차종/차명
                    combined_year,                              # 형식/연식
                    parsed_data.get('engine_type', ''),         # 원동기형식
                    parsed_data.get('owner_name', ''),          # 소유자
                    parsed_data.get('owner_address', ''),       # 주소
                    parsed_data.get('registration_date', ''),   # 최초등록일
                    parsed_data.get('vehicle_specs', ''),       # 제원관리번호
                    time.strftime("%Y-%m-%d %H:%M:%S")          # 처리시각
                ]

                # Upload to sheet
                if sheet_client.service:
                    sheet_client.append_row(row_data)

                print(f"  -> VehNo: {vehicle_no}, VIN: {vin}, Fuel: {parsed_data.get('fuel_type', 'Unknown')}")

                # Cleanup temp files
                if filename.lower().endswith('.pdf'):
                    ImagePreprocessor.cleanup_temp_files([image_path], filepath)

            processed_count += 1

        except Exception as e:
            print(f"  Error: {e}")
            error_count += 1

        # Garbage collection
        if (i + 1) % GC_INTERVAL == 0:
            gc.collect()
            print(f"  [GC] Collected after {i + 1} files")

        # Rate limiting
        time.sleep(RATE_LIMIT_SECONDS)

    # Final cleanup
    print(f"\nProcessing complete!")
    print(f"  Processed: {processed_count}")
    print(f"  Errors: {error_count}")
    print(f"  Last index: {start_idx + total_files - 1}")

    HybridOCREngine.cleanup()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Process Dropbox vehicle registration files')
    parser.add_argument('--file-list',
                        default='/Users/ssh/Documents/Develope/OCR_Multi_02/dropbox_registration_files.txt',
                        help='Path to file containing list of files to process')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting index for resuming (default: 0)')
    parser.add_argument('--max', type=int, default=None,
                        help='Maximum number of files to process (default: all)')
    parser.add_argument('--skip-clear', action='store_true',
                        help='Skip clearing the sheet (for resuming)')

    args = parser.parse_args()

    process_dropbox_files(
        file_list_path=args.file_list,
        start_idx=args.start,
        max_files=args.max,
        skip_sheet_clear=args.skip_clear
    )


if __name__ == "__main__":
    main()
