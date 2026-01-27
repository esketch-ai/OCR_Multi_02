# -*- coding: utf-8 -*-
"""
Debug script for OCR processing with memory optimization.
This version uses singleton pattern and avoids multiprocessing overhead.
"""
import os
import time
import gc
import signal
from dotenv import load_dotenv


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Processing timed out")


def process_single_file(filepath, engine, preprocessor, parser):
    """
    Process a single file with the shared engine instances.
    """
    try:
        # Process only FIRST page of PDF to save memory
        images = preprocessor.load_image(filepath, max_pages=1)
        if not images:
            return {'status': 'empty'}

        result_payload = {}

        if len(images) > 0:
            # Run Hybrid OCR on the first page
            res = engine.detect_text_hybrid(images[0])

            # Run Parser (pass filename for vehicle_no fallback)
            parsed_data = parser.parse_hybrid(res, filename=os.path.basename(filepath))

            # Add raw text for debugging if needed
            parsed_data['raw_text_google'] = res.get('google', {}).get('text', '')

            result_payload = parsed_data

            # Cleanup temp files immediately
            from src.ocr.preprocessor import ImagePreprocessor
            ImagePreprocessor.cleanup_temp_files(images, filepath)

        return {
            'filepath': filepath,
            'data': result_payload,
            'status': 'success'
        }

    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def debug_files_strict():
    load_dotenv()
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') is None:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service_account.json'

    # Initialize OCR components once (singleton pattern)
    print("Initializing OCR components...")
    from src.ocr.hybrid_engine import HybridOCREngine
    from src.ocr.preprocessor import ImagePreprocessor
    from src.parser.car_registration import CarRegistrationParser

    # Use Google Vision only for stability (no PaddleOCR)
    engine = HybridOCREngine(enable_paddle=False)
    preprocessor = ImagePreprocessor(dpi=150)  # Lower DPI for memory efficiency
    parser = CarRegistrationParser()
    print("OCR components initialized.")

    # Lazy import for Sheets
    from src.storage.gsheets import GoogleSheetClient

    # "상주" -> \uc0c1\uc8fc
    target_owner_part = "\uc0c1\uc8fc"

    SKIP_PARTIALS = ["1016"]

    # Init Google Sheet
    print("Initializing Google Sheet Client...")
    sheet_client = GoogleSheetClient()
    if sheet_client.service:
        print("Clearing existing Sheet content...")
        sheet_client.clear_first_sheet()
        # Add Header (Structured Fields)
        header = [
            "Filename",
            "Vehicle No", "VIN", "Owner Name", "Registration Date",
            "Model Name", "Model Year", "Vehicle Type", "Engine Type",
            "Found Time", "Match Status", "Confidence (VIN/Ref)"
        ]
        sheet_client.append_row(header)
    else:
        print("Warning: Google Sheet Client failed to initialize. Results will only be printed.")

    target_files = []
    # DIRECTORY UPDATE: Scan both 'processed' and 'Data'
    scan_dirs = ["processed", "Data"]
    print(f"Scanning directories: {scan_dirs}...")

    for scan_dir in scan_dirs:
        if not os.path.exists(scan_dir):
            print(f"Directory not found: {scan_dir}")
            continue

        for root, _, files in os.walk(scan_dir):
            for filename in files:
                if any(s in filename for s in SKIP_PARTIALS):
                    continue

                if filename.lower().endswith(('.jpg', '.png', '.pdf')):
                    filepath = os.path.join(root, filename)
                    target_files.append(filepath)

    total_files = len(target_files)
    print(f"Found {total_files} files to process.")

    # Set up timeout handler (Unix only)
    TIMEOUT_SECONDS = 60
    use_timeout = hasattr(signal, 'SIGALRM')
    if use_timeout:
        signal.signal(signal.SIGALRM, timeout_handler)

    # Memory management interval
    GC_INTERVAL = 10

    for i, filepath in enumerate(target_files):
        print(f"[{i+1}/{total_files}] Processing {os.path.basename(filepath)}...")

        try:
            # Set timeout (Unix only)
            if use_timeout:
                signal.alarm(TIMEOUT_SECONDS)

            res = process_single_file(filepath, engine, preprocessor, parser)

            # Cancel timeout
            if use_timeout:
                signal.alarm(0)

            if res['status'] == 'success':
                data = res.get('data', {})

                # Check Match Logic (Owner + Year/Date)
                raw_text = data.get('raw_text_google', '')

                is_match = False
                if raw_text and "2011" in raw_text and "11" in raw_text and "21" in raw_text:
                    if target_owner_part in raw_text:
                        is_match = True
                        print(">>> MATCH FOUND <<<")

                # Save to Sheet (Structured)
                if sheet_client.service:
                    try:
                        row = [
                            os.path.basename(filepath),
                            data.get('vehicle_no'),
                            data.get('vin'),
                            data.get('owner_name'),
                            data.get('registration_date'),
                            data.get('model_name'),
                            data.get('model_year'),
                            data.get('vehicle_type'),
                            data.get('engine_type'),
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            "MATCH" if is_match else "NO MATCH",
                            f"{data.get('confidences', {}).get('vin', 0):.2f}" if data.get('confidences') else "0.00"
                        ]
                        # Handle None values
                        row = [str(x) if x is not None else "" for x in row]

                        sheet_client.append_row(row)
                        print("Saved to sheet.")
                    except Exception as e:
                        print(f"Failed to append to sheet: {e}")

            elif res['status'] == 'error':
                print(f"Error: {res.get('error')}")
            elif res['status'] == 'empty':
                print("No images found in file.")

        except TimeoutError:
            print(f"!!! TIMEOUT: Processing took longer than {TIMEOUT_SECONDS}s. Skipping.")
            if use_timeout:
                signal.alarm(0)
        except Exception as e:
            print(f"Unexpected error: {e}")
            if use_timeout:
                signal.alarm(0)

        # Periodic garbage collection
        if (i + 1) % GC_INTERVAL == 0:
            gc.collect()
            print(f"[GC] Garbage collection performed after {i+1} files")

        # Rate limit to avoid API quota issues
        time.sleep(1.0)

    # Final cleanup
    print(f"\nProcessing complete. Total files: {total_files}")
    HybridOCREngine.cleanup()
    gc.collect()


if __name__ == "__main__":
    debug_files_strict()
