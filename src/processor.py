# -*- coding: utf-8 -*-
"""
Core processing pipeline for vehicle registration OCR.
Orchestrates: image preprocessing -> PaddleOCR -> parsing -> validation -> results.

Enhanced with OCR_Vehicle_02 algorithms:
- Image preprocessing (CLAHE, deskew, denoise)
- Coordinate-based field extraction (bbox Y%/X% zones)
- Standards-based validation (자동차관리법 규격)
- PP-Structure layout analysis (form-aware extraction + ensemble)
"""
import gc
import logging
import traceback
from src.ocr.paddle_engine import LocalPaddleEngine
from src.ocr.preprocessor import ImagePreprocessor
from src.parser.car_registration import CarRegistrationParser
from src.parser.form_parser import FormParser
from src.validator.vin_validator import VINValidator, decode_model_year
from src.validator.standards import (
    FUEL_TYPE_CORRECTIONS, DIMENSION_RANGES, UNIVERSAL_RANGES, NOISE_PATTERNS
)
from src.config import Config

logger = logging.getLogger(__name__)

# Module-level instances (initialized lazily)
_ocr_engine_ko = None
_ocr_engine_en = None
_layout_engine = None
_parser = CarRegistrationParser()
_form_parser = FormParser()
_validator = VINValidator()
_preprocessor = ImagePreprocessor(dpi=Config.PDF_DPI, max_size=Config.MAX_IMAGE_SIZE)

# Fields best recognized by English model (alphanumeric-heavy)
_EN_FIELDS = {'vin', 'engine_type', 'vehicle_format', 'length_mm', 'width_mm',
              'height_mm', 'total_weight_kg', 'passenger_capacity', 'model_year'}
# Fields best recognized by Korean model
_KO_FIELDS = {'vehicle_no', 'owner_name', 'model_name', 'vehicle_type',
              'fuel_type', 'registration_date', 'usage'}


def get_ocr_engine(lang=None):
    """Get or create the PaddleOCR engine singleton for given language."""
    global _ocr_engine_ko, _ocr_engine_en
    if lang == 'en':
        if _ocr_engine_en is None:
            _ocr_engine_en = LocalPaddleEngine(lang='en', enable_paddle=True)
        return _ocr_engine_en
    else:
        if _ocr_engine_ko is None:
            _ocr_engine_ko = LocalPaddleEngine(lang=Config.OCR_LANGUAGE, enable_paddle=True)
        return _ocr_engine_ko


def get_layout_engine():
    """Get or create the PP-Structure layout engine singleton."""
    global _layout_engine
    if _layout_engine is None and Config.ENABLE_LAYOUT_ANALYSIS:
        try:
            from src.ocr.layout_engine import LayoutEngine
            _layout_engine = LayoutEngine()
            if not _layout_engine.enabled:
                _layout_engine = None
        except Exception as e:
            logger.warning(f"Layout engine init failed, continuing without it: {e}")
            _layout_engine = None
    return _layout_engine


def warmup():
    """Pre-initialize OCR and layout engines at startup to avoid first-request timeout."""
    logger.info("Warming up OCR engines...")
    try:
        get_ocr_engine('korean')
        logger.info("PaddleOCR (korean) engine ready.")
    except Exception as e:
        logger.warning(f"PaddleOCR (korean) warmup failed: {e}")
    try:
        get_ocr_engine('en')
        logger.info("PaddleOCR (en) engine ready.")
    except Exception as e:
        logger.warning(f"PaddleOCR (en) warmup failed: {e}")
    try:
        get_layout_engine()
        logger.info("Layout engine ready.")
    except Exception as e:
        logger.warning(f"Layout engine warmup failed: {e}")
    logger.info("Warmup complete.")


def _correct_fuel_type_ocr(fuel_type):
    """Correct OCR misread fuel types using standards mapping."""
    if not fuel_type:
        return fuel_type

    # Direct correction
    if fuel_type in FUEL_TYPE_CORRECTIONS:
        corrected = FUEL_TYPE_CORRECTIONS[fuel_type]
        logger.info(f"Fuel type OCR correction: '{fuel_type}' → '{corrected}'")
        return corrected

    # Substring match
    for wrong, correct in FUEL_TYPE_CORRECTIONS.items():
        if wrong in fuel_type:
            logger.info(f"Fuel type OCR correction (substring): '{fuel_type}' → '{correct}'")
            return correct

    return fuel_type


def _validate_dimensions_by_type(parsed_data):
    """Validate dimensions against vehicle type standards (자동차관리법)."""
    vehicle_type = parsed_data.get('vehicle_type', '')

    # Normalize vehicle type (remove spaces)
    if vehicle_type:
        vehicle_type_clean = vehicle_type.replace(' ', '')
    else:
        vehicle_type_clean = ''

    # Get ranges for this vehicle type, fallback to universal
    ranges = DIMENSION_RANGES.get(vehicle_type_clean, UNIVERSAL_RANGES)

    field_map = {
        'length_mm': 'length_mm',
        'width_mm': 'width_mm',
        'height_mm': 'height_mm',
        'total_weight_kg': 'weight_kg',
        'passenger_capacity': 'capacity',
    }

    for data_field, range_key in field_map.items():
        value = parsed_data.get(data_field)
        if not value or range_key not in ranges:
            continue

        try:
            v = int(value)
            min_val, max_val = ranges[range_key]
            if v < min_val or v > max_val:
                logger.info(
                    f"Dimension out of range: {data_field}={v} "
                    f"(allowed {min_val}-{max_val} for '{vehicle_type_clean or 'universal'}')"
                )
                parsed_data[data_field] = None
        except (ValueError, TypeError):
            pass


def _extract_bbox_fields(ocr_results, img_height, img_width):
    """Extract fields using coordinate-based Y%/X% zone mapping.

    Zone layout from OCR_Vehicle_02 architecture:
        Y: 12-31% → Basic info (①-⑩)
        Y: 53-67% → Specifications table

    Returns dict of extracted fields (may be partial).
    """
    if not ocr_results or not img_height or not img_width:
        return {}

    def in_zone(r, y_min, y_max, x_min=0, x_max=100):
        """Check if OCR result bbox center is within percentage zone."""
        x1, y1, x2, y2 = r['bbox']
        cy = ((y1 + y2) / 2) / img_height * 100
        cx = ((x1 + x2) / 2) / img_width * 100
        return y_min <= cy <= y_max and x_min <= cx <= x_max

    def best_in_zone(y_min, y_max, x_min=0, x_max=100, min_conf=0.5):
        """Get highest-confidence text in a zone."""
        candidates = [
            r for r in ocr_results
            if in_zone(r, y_min, y_max, x_min, x_max)
            and r['confidence'] >= min_conf
            and not _is_noise(r['text'])
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r['confidence'])['text']

    result = {}

    # Basic info zone (Y: 14-28%)
    # ④차명 (Y: 16.5-19.5%, X: 17-42%)
    model_name = best_in_zone(16.5, 19.5, 17, 42)
    if model_name:
        result['model_name'] = model_name

    # ⑧소유자 (Y: 25-28.5%, X: 17-30%)
    owner = best_in_zone(25, 28.5, 17, 35)
    if owner:
        result['owner_name'] = owner

    # ②차종 (Y: 14-16.5%, X: 63-76%)
    vehicle_type = best_in_zone(14, 16.5, 63, 76)
    if vehicle_type:
        result['vehicle_type'] = vehicle_type

    return result


def _apply_layout_ensemble(parsed_data, layout_engine, image_path, img_h, img_w):
    """Apply PP-Structure layout analysis results to supplement text-based parsing.

    Only fills in fields that text-based parsing missed or has low confidence on.
    Layout engine failure does not affect the existing pipeline.
    """
    try:
        layout_result = layout_engine.analyze(image_path)
        if not layout_result.get('tables') and not layout_result.get('text_regions'):
            return

        form_fields = _form_parser.parse_layout(layout_result, img_h, img_w)

        filled = []
        for field_name, field_info in form_fields.items():
            value = field_info.get('value', '')
            if not value:
                continue

            existing = parsed_data.get(field_name)
            if not existing:
                parsed_data[field_name] = value
                filled.append(field_name)

        if filled:
            logger.info(f"Layout ensemble filled {len(filled)} fields: {filled}")

    except Exception as e:
        logger.warning(f"Layout ensemble failed (non-fatal): {e}")


def _is_noise(text):
    """Check if text is noise (legal/warning text)."""
    for pattern in NOISE_PATTERNS:
        if pattern in text:
            return True
    return False


def _get_image_dimensions(image_path):
    """Get image dimensions for coordinate-based extraction."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None, None


def _merge_dual_ocr(parsed_ko, parsed_en):
    """Merge Korean and English OCR results by field type.

    Korean engine is better for: 한글 라벨, 업체명, 차명, 차종, 연료
    English engine is better for: VIN, 엔진코드, 치수(숫자), 중량

    Strategy: For alphanumeric fields, prefer English result if it looks valid.
    For Korean text fields, prefer Korean result.
    """
    merged = dict(parsed_ko)

    for field in _EN_FIELDS:
        en_val = parsed_en.get(field)
        ko_val = parsed_ko.get(field)
        if not en_val:
            continue
        if not ko_val:
            merged[field] = en_val
            logger.info(f"Field '{field}' filled by EN engine: {en_val}")
            continue

        # For VIN: prefer the one that validates
        if field == 'vin':
            from src.validator.vin_validator import is_valid_structure
            en_valid, _ = is_valid_structure(en_val)
            ko_valid, _ = is_valid_structure(ko_val)
            if en_valid and not ko_valid:
                merged[field] = en_val
                logger.info(f"VIN replaced by EN engine: {ko_val} → {en_val}")
            continue

        # For numeric fields: prefer the one with more digits
        if field in ('length_mm', 'width_mm', 'height_mm', 'total_weight_kg', 'passenger_capacity'):
            import re
            en_digits = len(re.sub(r'\D', '', en_val))
            ko_digits = len(re.sub(r'\D', '', ko_val))
            if en_digits > ko_digits:
                merged[field] = en_val
                logger.info(f"Field '{field}' replaced by EN engine: {ko_val} → {en_val}")

    return merged


def process_single_file(file_path, filename):
    """Process a single image/PDF file through the dual OCR pipeline."""
    try:
        engine_ko = get_ocr_engine('korean')

        # 1. Preprocess image/PDF (now with CLAHE, deskew, denoise)
        image_paths = _preprocessor.load_image(file_path)
        if not image_paths:
            return {'status': 'error', 'filename': filename, 'data': {}, 'message': 'Failed to load image'}

        image_path = image_paths[0]

        try:
            # 2a. Run Korean OCR (primary - for Korean labels and text)
            logger.info(f"Running Korean OCR on: {image_path}")
            ocr_result_ko = engine_ko.detect_text(image_path)
            ocr_text = ocr_result_ko.get('text', '')
            ocr_results = ocr_result_ko.get('ocr_results', [])
            logger.info(f"Korean OCR: text={len(ocr_text)} chars, bbox={len(ocr_results)}")

            if not ocr_text:
                debug = ocr_result_ko.get('debug', '')
                msg = f'No text detected [API:{engine_ko._api_version}, debug:{debug}]'
                return {'status': 'error', 'filename': filename, 'data': {}, 'message': msg}

            # 3. Verify document type
            if not _parser.verify_document_type(ocr_text):
                preview = ocr_text[:200].replace('\n', ' | ')
                return {'status': 'skipped', 'filename': filename, 'data': {},
                        'message': f'Not a vehicle registration certificate. OCR preview: {preview}'}

            # 4a. Parse Korean OCR text
            parsed_ko = _parser.parse_single(ocr_text, filename=filename)

            # 2b. Run English OCR (secondary - for VIN, codes, numbers)
            engine_en = get_ocr_engine('en')
            if engine_en and engine_en.enabled:
                try:
                    logger.info(f"Running English OCR on: {image_path}")
                    ocr_result_en = engine_en.detect_text(image_path)
                    ocr_text_en = ocr_result_en.get('text', '')
                    ocr_results_en = ocr_result_en.get('ocr_results', [])
                    logger.info(f"English OCR: text={len(ocr_text_en)} chars, bbox={len(ocr_results_en)}")

                    if ocr_text_en:
                        # 4b. Parse English OCR text
                        parsed_en = _parser.parse_single(ocr_text_en, filename=filename)
                        # 4c. Merge: field-type-aware selection
                        parsed_data = _merge_dual_ocr(parsed_ko, parsed_en)

                        # Use English bbox results too (better for alphanumeric)
                        if ocr_results_en:
                            ocr_results = ocr_results or ocr_results_en
                    else:
                        parsed_data = parsed_ko
                except Exception as e:
                    logger.warning(f"English OCR failed (non-fatal): {e}")
                    parsed_data = parsed_ko
            else:
                parsed_data = parsed_ko

            # 5. Coordinate-based extraction (supplement text-based results)
            img_w, img_h = _get_image_dimensions(image_path)
            if ocr_results:
                if img_w and img_h:
                    bbox_fields = _extract_bbox_fields(ocr_results, img_h, img_w)
                    for field, value in bbox_fields.items():
                        if not parsed_data.get(field):
                            parsed_data[field] = value
                            logger.info(f"Field '{field}' filled by bbox extraction: {value}")

            # 5.5. PP-Structure layout analysis (ensemble with text-based results)
            layout_engine = get_layout_engine()
            if layout_engine:
                _apply_layout_ensemble(parsed_data, layout_engine, image_path, img_h, img_w)

            # 6. Apply fuel type OCR corrections (자동차관리법)
            if parsed_data.get('fuel_type'):
                parsed_data['fuel_type'] = _correct_fuel_type_ocr(parsed_data['fuel_type'])

            # 7. Validate dimensions against vehicle type standards
            _validate_dimensions_by_type(parsed_data)

            # 8. Validate VIN
            vin = parsed_data.get('vin')
            is_valid, validation_msg = _validator.validate(vin)
            parsed_data['vin_valid'] = is_valid
            parsed_data['vin_message'] = validation_msg

            # 9. Decode model year from VIN if not already extracted
            if vin and not parsed_data.get('model_year'):
                vin_year = decode_model_year(vin)
                if vin_year:
                    parsed_data['model_year'] = str(vin_year)
                    logger.info(f"Model year from VIN: {vin_year}")

            # Include OCR text preview for debugging
            parsed_data['_ocr_preview'] = ocr_text[:300].replace('\n', ' | ')

            logger.info(f"Successfully processed: {filename}")
            return {'status': 'success', 'filename': filename, 'data': parsed_data, 'message': 'OK'}

        finally:
            # Cleanup temp files
            ImagePreprocessor.cleanup_temp_files(image_paths, file_path)

    except Exception as e:
        logger.error(f"Error processing {filename}: {e}\n{traceback.format_exc()}")
        return {'status': 'error', 'filename': filename, 'data': {}, 'message': str(e)}


def process_batch(file_list, progress_callback=None):
    """Process a batch of files."""
    results = []
    total = len(file_list)

    for i, (file_path, filename) in enumerate(file_list):
        result = process_single_file(file_path, filename)
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total)

        if (i + 1) % Config.GC_INTERVAL == 0:
            gc.collect()

    return results


def results_to_rows(results):
    """Convert processing results to rows for Excel output."""
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
