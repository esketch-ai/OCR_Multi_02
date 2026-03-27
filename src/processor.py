# -*- coding: utf-8 -*-
"""
Core processing pipeline for vehicle registration OCR.
Orchestrates: image preprocessing -> PaddleOCR -> parsing -> validation -> results.
"""
import gc
import logging
import traceback
from src.ocr.paddle_engine import LocalPaddleEngine
from src.ocr.preprocessor import ImagePreprocessor
from src.parser.car_registration import CarRegistrationParser
from src.validator.vin_validator import VINValidator
from src.config import Config

logger = logging.getLogger(__name__)

# Module-level instances (initialized lazily)
_ocr_engine = None
_parser = CarRegistrationParser()
_validator = VINValidator()
_preprocessor = ImagePreprocessor(dpi=Config.PDF_DPI, max_size=Config.MAX_IMAGE_SIZE)


def get_ocr_engine():
    """Get or create the PaddleOCR engine singleton."""
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = LocalPaddleEngine(lang=Config.OCR_LANGUAGE, enable_paddle=True)
    return _ocr_engine


def process_single_file(file_path, filename):
    """
    Process a single image/PDF file through the OCR pipeline.

    Args:
        file_path: Absolute path to the file
        filename: Original filename (used for fallback extraction)

    Returns:
        dict with keys: status, filename, data, message
    """
    try:
        engine = get_ocr_engine()

        # 1. Preprocess image/PDF
        image_paths = _preprocessor.load_image(file_path)
        if not image_paths:
            return {'status': 'error', 'filename': filename, 'data': {}, 'message': 'Failed to load image'}

        # Process first page only (vehicle registration is single page)
        image_path = image_paths[0]

        try:
            # 2. Run PaddleOCR
            logger.info(f"Running OCR on: {image_path}")
            ocr_result = engine.detect_text(image_path)
            ocr_text = ocr_result.get('text', '')
            logger.info(f"OCR text length: {len(ocr_text)}, preview: {ocr_text[:100] if ocr_text else 'EMPTY'}")

            if not ocr_text:
                debug = ocr_result.get('debug', '')
                msg = f'No text detected [API:{engine._api_version}, debug:{debug}]'
                return {'status': 'error', 'filename': filename, 'data': {}, 'message': msg}

            # 3. Verify document type
            if not _parser.verify_document_type(ocr_text):
                preview = ocr_text[:200].replace('\n', ' | ')
                return {'status': 'skipped', 'filename': filename, 'data': {}, 'message': f'Not a vehicle registration certificate. OCR preview: {preview}'}

            # 4. Parse
            parsed_data = _parser.parse_single(ocr_text, filename=filename)

            # 5. Validate VIN
            vin = parsed_data.get('vin')
            is_valid, validation_msg = _validator.validate(vin)
            parsed_data['vin_valid'] = is_valid
            parsed_data['vin_message'] = validation_msg

            # Include OCR text preview for debugging
            parsed_data['_ocr_preview'] = ocr_text[:300].replace('\n', ' | ')

            logger.info(f"Successfully processed: {filename}")
            return {'status': 'success', 'filename': filename, 'data': parsed_data, 'message': 'OK'}

        finally:
            # Cleanup temp files from PDF conversion
            if filename.lower().endswith('.pdf'):
                ImagePreprocessor.cleanup_temp_files(image_paths, file_path)

    except Exception as e:
        logger.error(f"Error processing {filename}: {e}\n{traceback.format_exc()}")
        return {'status': 'error', 'filename': filename, 'data': {}, 'message': str(e)}


def process_batch(file_list, progress_callback=None):
    """
    Process a batch of files.

    Args:
        file_list: List of (file_path, filename) tuples
        progress_callback: Optional callable(current, total) for progress updates

    Returns:
        List of result dicts from process_single_file
    """
    results = []
    total = len(file_list)

    for i, (file_path, filename) in enumerate(file_list):
        result = process_single_file(file_path, filename)
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total)

        # Memory management
        if (i + 1) % Config.GC_INTERVAL == 0:
            gc.collect()

    return results


def results_to_rows(results):
    """
    Convert processing results to rows for Excel output.

    Returns:
        List of lists, each inner list is one Excel row (13 columns)
    """
    rows = []
    for r in results:
        if r['status'] != 'success':
            continue
        d = r['data']
        rows.append([
            d.get('vehicle_no', ''),
            d.get('owner_name', ''),
            d.get('vin', ''),
            d.get('model_name', ''),
            d.get('model_year', ''),
            d.get('registration_date', ''),
            d.get('vehicle_type', ''),
            d.get('length_mm', ''),
            d.get('width_mm', ''),
            d.get('height_mm', ''),
            d.get('total_weight_kg', ''),
            d.get('passenger_capacity', ''),
            d.get('fuel_type', ''),
        ])
    return rows
