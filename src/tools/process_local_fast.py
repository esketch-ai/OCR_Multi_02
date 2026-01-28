# -*- coding: utf-8 -*-
"""
Fast parallel processing of vehicle registration certificates using LOCAL OCR.
Uses multiprocessing for parallel OCR processing.
"""
import os
import sys
import time
import shutil
import argparse
import unicodedata
import logging
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count, set_start_method

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def normalize_path(path):
    """Normalize Unicode path for consistent handling."""
    return unicodedata.normalize('NFC', path)


def sanitize_filename(text):
    """Remove invalid characters from filename."""
    if not text:
        return ''
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        text = text.replace(char, '_')
    return text.strip()


def init_worker():
    """Initialize OCR engine in worker process."""
    import warnings
    import os
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['GLOG_minloglevel'] = '3'
    # Suppress PaddleOCR logging
    import logging
    logging.getLogger('ppocr').setLevel(logging.ERROR)


def process_single_file(args):
    """
    Process a single file. Called by worker processes.

    Args:
        args: tuple of (filepath, output_dir, dpi)

    Returns:
        dict with processing results
    """
    filepath, output_dir, dpi = args
    filename = os.path.basename(filepath)

    try:
        # Import inside function to avoid multiprocessing issues
        import warnings
        import os
        import logging as log
        warnings.filterwarnings('ignore')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['GLOG_minloglevel'] = '3'
        log.getLogger('ppocr').setLevel(log.ERROR)

        from paddleocr import PaddleOCR
        from PIL import Image
        import tempfile
        import re

        # Initialize OCR for this worker (lazy initialization)
        # Balanced configuration: mobile detection + Korean recognition
        # Higher det_limit for better accuracy on registration certificates
        ocr = PaddleOCR(
            lang='korean',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name='PP-OCRv5_mobile_det',
            text_recognition_model_name='korean_PP-OCRv5_mobile_rec',
            text_det_limit_side_len=1920,  # Higher for better accuracy
            text_det_limit_type='max',
            text_det_box_thresh=0.5,       # Lower threshold to catch more text
        )

        # Import parser
        from src.parser.car_registration import CarRegistrationParser
        from src.validator.vin_validator import VINValidator

        parser = CarRegistrationParser()
        vin_validator = VINValidator()

        # Check if file exists
        if not os.path.exists(filepath):
            return {'status': 'skip', 'reason': 'file_not_found', 'filename': filename}

        # Handle PDF vs Image
        if filepath.lower().endswith('.pdf'):
            from pdf2image import convert_from_path
            pages = convert_from_path(filepath, dpi=dpi, first_page=1, last_page=1)
            if not pages:
                return {'status': 'skip', 'reason': 'pdf_empty', 'filename': filename}

            # Save to temp file
            temp_path = os.path.join(tempfile.gettempdir(), f"{os.path.splitext(filename)[0]}_p1.jpg")
            pages[0].save(temp_path, 'JPEG', quality=85)
            pages[0].close()
            image_path = temp_path
        else:
            # Check if image needs resizing
            with Image.open(filepath) as img:
                width, height = img.size
                if width > 2048 or height > 2048:
                    # Resize
                    if width > height:
                        new_width = 2048
                        new_height = int(height * (2048 / width))
                    else:
                        new_height = 2048
                        new_width = int(width * (2048 / height))

                    resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    temp_path = os.path.join(tempfile.gettempdir(), f"{os.path.splitext(filename)[0]}_resized.jpg")
                    resized.save(temp_path, 'JPEG', quality=85)
                    resized.close()
                    image_path = temp_path
                else:
                    image_path = filepath

        # Run OCR
        result = ocr.predict(image_path)
        res = result[0].json['res']
        texts = res.get('rec_texts', [])
        scores = res.get('rec_scores', [])

        if not texts:
            return {'status': 'skip', 'reason': 'no_text', 'filename': filename}

        full_text = '\n'.join(texts)

        # Verify document type
        if not parser.verify_document_type(full_text):
            return {'status': 'skip', 'reason': 'not_registration', 'filename': filename}

        # Create hybrid result format for parser compatibility
        raw_result = []
        boxes = res.get('rec_polys', [])
        for i, (text, score) in enumerate(zip(texts, scores)):
            box = boxes[i] if i < len(boxes) else [[0, 0], [0, 0], [0, 0], [0, 0]]
            raw_result.append([box, (text, score)])

        hybrid_result = {
            'google': {
                'text': full_text,
                'avg_conf': sum(scores) / len(scores) if scores else 0.0,
                'annotation': None
            },
            'paddle': {
                'result': raw_result,
                'text_lines': texts
            }
        }

        # Parse data
        parsed_data = parser.parse_hybrid(hybrid_result, filename=filename)
        confidences = parsed_data.get('confidences', {})

        # Validate VIN
        vin = parsed_data.get('vin')
        vehicle_no = parsed_data.get('vehicle_no')
        is_valid, validation_msg = vin_validator.validate(vin)

        fuel_type = parsed_data.get('fuel_type', 'Unknown')

        # Copy file to output
        copied_path = None
        if vehicle_no and output_dir:
            vehicle_no_clean = sanitize_filename(vehicle_no)
            fuel_type_clean = sanitize_filename(fuel_type) if fuel_type else 'Unknown'
            _, ext = os.path.splitext(filepath)
            new_filename = f"{vehicle_no_clean}_{fuel_type_clean}{ext}"

            os.makedirs(output_dir, exist_ok=True)
            dest_path = os.path.join(output_dir, new_filename)

            if os.path.exists(dest_path):
                timestamp = datetime.now().strftime('%H%M%S')
                new_filename = f"{vehicle_no_clean}_{fuel_type_clean}_{timestamp}{ext}"
                dest_path = os.path.join(output_dir, new_filename)

            shutil.copy2(filepath, dest_path)
            copied_path = new_filename

        # Cleanup temp files
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            'status': 'success',
            'filename': filename,
            'vehicle_no': vehicle_no,
            'vin': vin,
            'vin_valid': is_valid,
            'fuel_type': fuel_type,
            'copied_file': copied_path,
            'conf_veh': confidences.get('vehicle_no', 0.0),
            'conf_vin': confidences.get('vin', 0.0),
            'model_name': parsed_data.get('model_name', ''),
            'owner_name': parsed_data.get('owner_name', ''),
            'registration_date': parsed_data.get('registration_date', ''),
        }

    except Exception as e:
        return {'status': 'error', 'filename': filename, 'error': str(e)}


def process_files_parallel(file_list_path, output_dir, start_idx=0, max_files=None,
                           num_workers=None, dpi=100):
    """
    Process files in parallel using multiprocessing.
    """
    # Read file list
    with open(file_list_path, 'r', encoding='utf-8') as f:
        all_files = [normalize_path(line.strip()) for line in f if line.strip()]

    # Apply start index and max files
    files_to_process = all_files[start_idx:]
    if max_files:
        files_to_process = files_to_process[:max_files]

    total_files = len(files_to_process)

    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 4)  # Limit to 4 for memory reasons

    logger.info(f"Processing {total_files} files with {num_workers} workers")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"DPI: {dpi}")

    # Prepare arguments for each file
    args_list = [(fp, output_dir, dpi) for fp in files_to_process]

    # Initialize Google Sheet for results
    from src.storage.gsheets import GoogleSheetClient
    sheet_client = GoogleSheetClient()

    if sheet_client.service:
        logger.info("Clearing sheet and adding header...")
        sheet_client.clear_first_sheet()
        header = [
            "파일명", "차량번호", "차량번호 신뢰도", "차대번호(VIN)", "VIN 신뢰도",
            "VIN 유효성", "연료타입", "차명", "소유자", "최초등록일", "복사파일", "처리시각"
        ]
        sheet_client.append_row(header)

    # Process in parallel
    start_time = time.time()
    success_count = 0
    skip_count = 0
    error_count = 0

    # Use spawn to avoid fork issues with PaddleOCR
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    logger.info("Starting parallel processing...")

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results = pool.imap_unordered(process_single_file, args_list, chunksize=1)

        for i, result in enumerate(results):
            status = result.get('status')
            filename = result.get('filename', 'unknown')

            if status == 'success':
                success_count += 1
                logger.info(f"[{i+1}/{total_files}] ✓ {filename} -> {result.get('vehicle_no')}, {result.get('vin')}")

                # Add to sheet
                if sheet_client.service:
                    row = [
                        filename,
                        result.get('vehicle_no', ''),
                        f"{result.get('conf_veh', 0):.4f}",
                        result.get('vin', ''),
                        f"{result.get('conf_vin', 0):.4f}",
                        "Valid" if result.get('vin_valid') else "Invalid",
                        result.get('fuel_type', ''),
                        result.get('model_name', ''),
                        result.get('owner_name', ''),
                        result.get('registration_date', ''),
                        result.get('copied_file', ''),
                        time.strftime("%Y-%m-%d %H:%M:%S")
                    ]
                    sheet_client.append_row(row)

            elif status == 'skip':
                skip_count += 1
                logger.info(f"[{i+1}/{total_files}] - {filename}: SKIP ({result.get('reason')})")
            else:
                error_count += 1
                logger.warning(f"[{i+1}/{total_files}] ✗ {filename}: {result.get('error')}")

    elapsed = time.time() - start_time
    avg_time = elapsed / total_files if total_files > 0 else 0

    logger.info(f"\n{'='*50}")
    logger.info(f"Processing complete!")
    logger.info(f"  Total files: {total_files}")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Skipped: {skip_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Total time: {elapsed:.1f}s")
    logger.info(f"  Avg time per file: {avg_time:.1f}s")
    logger.info(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description='Fast parallel processing of vehicle registration files'
    )
    parser.add_argument('--file-list',
                        default='/Users/ssh/Documents/Develope/OCR_Multi_02/dropbox_registration_files.txt',
                        help='Path to file list')
    parser.add_argument('--output-dir',
                        default='/Users/ssh/Documents/Develope/OCR_Multi_02/output',
                        help='Output directory for copied files')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting index')
    parser.add_argument('--max', type=int, default=None,
                        help='Maximum files to process')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: auto)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for PDF conversion (default: 150)')

    args = parser.parse_args()

    process_files_parallel(
        file_list_path=args.file_list,
        output_dir=args.output_dir,
        start_idx=args.start,
        max_files=args.max,
        num_workers=args.workers,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
