import os
import sys
import shutil
import logging
import argparse
import time
import gc
from src.config import Config
from src.ocr.hybrid_engine import HybridOCREngine
from src.ocr.preprocessor import ImagePreprocessor
from src.parser.car_registration import CarRegistrationParser
from src.validator.vin_validator import VINValidator
from src.storage.gsheets import GoogleSheetClient
from src.storage.vin_logger import VinLogger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(Config.BASE_DIR, 'ocr_process.log'), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Memory management constants
GC_INTERVAL = 10  # Run garbage collection every N files
BATCH_SIZE = 50   # Process files in batches to manage memory

def main():
    parser_args = argparse.ArgumentParser(description='Vehicle Registration OCR System (Hybrid Expert Mode)')
    parser_args.add_argument('--input_dir', default=Config.INPUT_DIR, help='Root input directory to scan recursively')
    parser_args.add_argument('--no-paddle', action='store_true', help='Disable PaddleOCR (use Google Vision only)')
    parser_args.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Number of files to process per batch')
    parser_args.add_argument('--dpi', type=int, default=150, help='DPI for PDF conversion (default: 150)')
    args = parser_args.parse_args()

    input_dir = args.input_dir
    enable_paddle = not args.no_paddle
    batch_size = args.batch_size

    logger.info(f"Starting Vehicle Registration OCR System (Hybrid Recursive)")
    logger.info(f"Scanning directory: {input_dir}")
    logger.info(f"PaddleOCR enabled: {enable_paddle}, Batch size: {batch_size}, DPI: {args.dpi}")

    # Initialize components
    try:
        # Hybrid Engine with optional PaddleOCR
        ocr_engine = HybridOCREngine(enable_paddle=enable_paddle)

        parser_logic = CarRegistrationParser()
        preprocessor = ImagePreprocessor(dpi=args.dpi)
        vin_validator = VINValidator()
        sheet_client = GoogleSheetClient()
        vin_logger = VinLogger()
        logger.info("All components initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return

    # Recursive file discovery
    file_list = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf')):
                file_list.append(os.path.join(root, file))

    logger.info(f"Found {len(file_list)} files in {input_dir}")

    processed_count = 0
    for file_path in file_list:
        # Rate limiting to prevent 429 Errors
        time.sleep(1.5)

        try:
            # Determine relative path for structure preservation
            rel_path = os.path.relpath(file_path, input_dir)
            filename = os.path.basename(file_path)
            
            logger.info(f"Processing: {rel_path}")

            # 1. Preprocess
            image_paths = preprocessor.load_image(file_path)
            if not image_paths:
                logger.warning(f"Skipping {rel_path}: Invalid image or PDF not supported.")
                continue

            for idx, image_path in enumerate(image_paths):
                logger.info(f"Processing page {idx+1} of {filename}")
                
                # 2. Hybrid OCR
                # Returns dict with collected results
                hybrid_result = ocr_engine.detect_text_hybrid(image_path)
                
                # Use Google text for document type verification (Primary)
                g_text = hybrid_result['google']['text']
                if not g_text:
                    logger.warning(f"No text detected in {filename} page {idx+1}")
                    continue

                # 3. Verify Document Type
                if not parser_logic.verify_document_type(g_text):
                    logger.warning(f"Skipping {filename} page {idx+1}: Not a Vehicle Registration Certificate")
                    if filename.lower().endswith('.pdf') and os.path.exists(image_path):
                         os.remove(image_path)
                    continue
                    
                # 4. Parse Hybrid
                # Returns result dict with 'confidences' key
                # Pass filename for vehicle_no fallback extraction
                parsed_data = parser_logic.parse_hybrid(hybrid_result, filename=filename)
                confidences = parsed_data.get('confidences', {})
                logger.info(f"Extracted data (Hybrid): {parsed_data}")
                
                # 5. Validate VIN
                vin = parsed_data.get('vin')
                vehicle_no = parsed_data.get('vehicle_no')
                is_valid, validation_msg = vin_validator.validate(vin)
                
                # Granular Confidences
                conf_veh = confidences.get('vehicle_no', 0.0)
                conf_vin = confidences.get('vin', 0.0)
                
                # 6. Log VIN Stats
                vin_logger.log(f"{rel_path}_p{idx+1}", vehicle_no, vin, is_valid, validation_msg, conf_vin)
                
                # 7. Upload to Sheet
                vehicle_type = parsed_data.get('vehicle_type', '')
                model_name = parsed_data.get('model_name', '')
                combined_name = f"{vehicle_type} / {model_name}" if vehicle_type else model_name
                
                vehicle_format = parsed_data.get('vehicle_format', '')
                model_year = parsed_data.get('model_year', '')
                combined_year = f"{vehicle_format} / {model_year}" if vehicle_format else model_year

                row_data = [
                    parsed_data.get('vehicle_no', ''),      # 1: 차량번호
                    f"{conf_veh:.4f}",                      # 2: 차량번호 신뢰도
                    parsed_data.get('vin', ''),             # 3: 차대번호 (VIN)
                    f"{conf_vin:.4f}",                      # 4: 차대번호 신뢰도
                    parsed_data.get('fuel_type', 'Unknown'),# 5: 연료 타입 (NEW)
                    combined_name,                          # 6: 차종/차명
                    combined_year,                          # 7: 형식/연식
                    parsed_data.get('engine_type', ''),     # 8: 원동기형식
                    parsed_data.get('owner_name', ''),      # 9: 소유자
                    parsed_data.get('owner_address', ''),   # 10: 주소
                    parsed_data.get('registration_date', ''),# 11: 최초등록일
                    parsed_data.get('vehicle_specs', '')    # 12: 제원관리번호
                ]
                sheet_client.append_row(row_data)
                
                # Cleanup temp PDF page
                if filename.lower().endswith('.pdf'):
                    ImagePreprocessor.cleanup_temp_files([image_path], file_path)
            
            # 8. Move original file to Processed
            target_dir = os.path.join(Config.PROCESSED_DIR, os.path.dirname(rel_path))
            os.makedirs(target_dir, exist_ok=True)

            if os.path.exists(file_path):
                shutil.move(file_path, os.path.join(target_dir, filename))
                logger.info(f"Successfully processed {rel_path} -> {target_dir}")

            # Memory management: cleanup after each file
            processed_count += 1
            if processed_count % GC_INTERVAL == 0:
                gc.collect()
                logger.debug(f"Garbage collection performed after {processed_count} files")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            try:
                if os.path.exists(file_path):
                    target_dir = os.path.join(Config.FAILED_DIR, os.path.dirname(rel_path))
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.move(file_path, os.path.join(target_dir, filename))
            except Exception as move_error:
                logger.error(f"Failed to move file {filename} to failed dir: {move_error}")

    # Final cleanup
    logger.info(f"Processing complete. Total files processed: {processed_count}")
    HybridOCREngine.cleanup()
    gc.collect()

if __name__ == "__main__":
    main()
