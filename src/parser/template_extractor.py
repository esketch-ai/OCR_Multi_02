# -*- coding: utf-8 -*-
"""
Template-based field extractor for Korean Vehicle Registration Certificate.
Uses learned field positions to crop and OCR specific regions.
"""
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FieldRegion:
    """Defines a field region in normalized coordinates (0-1)."""
    name: str           # Field name
    korean: str         # Korean label
    x1: float          # Left (normalized)
    y1: float          # Top (normalized)
    x2: float          # Right (normalized)
    y2: float          # Bottom (normalized)
    value_offset_x: float = 0.05  # X offset from marker to value
    value_offset_y: float = 0.0   # Y offset from marker to value


class TemplateExtractor:
    """
    Template-based extractor for vehicle registration certificates.

    The document has a fixed layout where circled numbers ①-⑩ mark field labels,
    and the corresponding values appear to the right or below.

    Normalized coordinates based on standard document layout:
    - Document is typically 3508 x 2481 pixels (A4 landscape at 300 DPI)
    - Top section (0.15 - 0.45 Y): Main registration info
    - Middle section (0.45 - 0.75 Y): Specifications (제원)
    - Bottom section (0.75 - 1.0 Y): Other info
    """

    # Template regions based on document analysis
    # Format: FieldRegion(name, korean, x1, y1, x2, y2)
    # Coordinates are normalized (0-1)

    MAIN_FIELDS = [
        # Row 1: ① 자동차등록번호, ② 차종, ③ 용도
        FieldRegion('vehicle_no', '자동차등록번호', 0.12, 0.20, 0.42, 0.26),
        FieldRegion('vehicle_type', '차종', 0.45, 0.20, 0.65, 0.26),
        FieldRegion('usage_type', '용도', 0.68, 0.20, 0.95, 0.26),

        # Row 2: ④ 차명, ⑤ 형식 및 연식
        FieldRegion('model_name', '차명', 0.12, 0.24, 0.42, 0.30),
        FieldRegion('vehicle_format', '형식', 0.45, 0.24, 0.65, 0.30),
        FieldRegion('model_year', '모델연도', 0.68, 0.24, 0.95, 0.30),

        # Row 3: ⑥ 차대번호, ⑦ 원동기형식
        FieldRegion('vin', '차대번호', 0.12, 0.28, 0.55, 0.34),
        FieldRegion('engine_type', '원동기형식', 0.55, 0.28, 0.95, 0.34),

        # Row 4: ⑧ 성명(명칭), ⑨ 주민등록번호
        FieldRegion('owner_name', '성명', 0.12, 0.32, 0.42, 0.38),
        FieldRegion('owner_id', '주민등록번호', 0.45, 0.32, 0.95, 0.38),

        # Row 5: ⑩ 사용본거지
        FieldRegion('owner_address', '사용본거지', 0.12, 0.36, 0.95, 0.42),
    ]

    SPEC_FIELDS = [
        # 제원 Section (1. 제원)
        # Usually starts around Y = 0.48
        FieldRegion('spec_no', '제원관리번호', 0.12, 0.48, 0.40, 0.52),
        FieldRegion('length_mm', '길이', 0.12, 0.52, 0.25, 0.56),
        FieldRegion('width_mm', '너비', 0.25, 0.52, 0.38, 0.56),
        FieldRegion('height_mm', '높이', 0.38, 0.52, 0.50, 0.56),
        FieldRegion('total_weight_kg', '총중량', 0.12, 0.56, 0.25, 0.60),
        FieldRegion('displacement_cc', '배기량', 0.12, 0.60, 0.25, 0.64),
        FieldRegion('rated_output', '정격출력', 0.25, 0.60, 0.40, 0.64),
        FieldRegion('passenger_capacity', '승차정원', 0.40, 0.56, 0.55, 0.60),
        FieldRegion('max_load_kg', '최대적재량', 0.55, 0.56, 0.70, 0.60),
        FieldRegion('fuel_type', '연료의종류', 0.25, 0.64, 0.50, 0.68),
    ]

    # Additional fields
    EXTRA_FIELDS = [
        FieldRegion('first_reg_date', '최초등록일', 0.60, 0.15, 0.95, 0.20),
    ]

    # Fuel type mappings
    FUEL_TYPES = {
        '전기': 'Electric',
        'CNG': 'CNG',
        '천연가스': 'CNG',
        '디젤': 'Diesel',
        '경유': 'Diesel',
        'LPG': 'LPG',
        '휘발유': 'Gasoline',
        '수소': 'Hydrogen',
        '하이브리드': 'Hybrid',
    }

    def __init__(self):
        self.all_fields = self.MAIN_FIELDS + self.SPEC_FIELDS + self.EXTRA_FIELDS

    def extract_from_image(self, image_path: str, ocr_engine) -> Dict[str, Any]:
        """
        Extract fields from image using template regions.

        Args:
            image_path: Path to the image file
            ocr_engine: Initialized PaddleOCR instance

        Returns:
            Dictionary with extracted fields
        """
        from PIL import Image
        import numpy as np

        # Load image
        img = Image.open(image_path)
        width, height = img.size

        result = {}
        confidences = {}

        # Extract each field region
        for field in self.all_fields:
            # Calculate pixel coordinates
            x1 = int(field.x1 * width)
            y1 = int(field.y1 * height)
            x2 = int(field.x2 * width)
            y2 = int(field.y2 * height)

            # Crop region
            region = img.crop((x1, y1, x2, y2))

            # Convert to numpy for OCR
            region_np = np.array(region)

            # Run OCR on region
            try:
                ocr_result = ocr_engine.predict(region_np)
                if ocr_result and ocr_result[0]:
                    res = ocr_result[0].json.get('res', {})
                    texts = res.get('rec_texts', [])
                    scores = res.get('rec_scores', [])

                    if texts:
                        # Combine all text in region
                        raw_text = ' '.join(texts)
                        # Clean and extract value
                        value = self._clean_field_value(raw_text, field.name, field.korean)

                        if value:
                            result[field.name] = value
                            confidences[field.name] = sum(scores) / len(scores) if scores else 0.5
            except Exception as e:
                logger.debug(f"Failed to extract {field.name}: {e}")
                continue

        # Post-process specific fields
        result = self._post_process(result)
        result['confidences'] = confidences

        return result

    def extract_from_boxes(self, ocr_boxes: List, image_width: int, image_height: int) -> Dict[str, Any]:
        """
        Extract fields using marker-based approach.
        Finds circled number markers first, then extracts values relative to them.

        Args:
            ocr_boxes: List of (box, (text, score)) from PaddleOCR
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            Dictionary with extracted fields
        """
        result = {}
        confidences = {}

        # Index all boxes with position info
        indexed_boxes = []
        for item in ocr_boxes:
            if not item or len(item) < 2:
                continue
            box = item[0]
            text_data = item[1]

            if isinstance(text_data, tuple):
                text, score = text_data[0], text_data[1]
            else:
                text, score = str(text_data), 0.5

            if not text.strip():
                continue

            xs = [p[0] for p in box]
            ys = [p[1] for p in box]

            indexed_boxes.append({
                'text': text.strip(),
                'score': score,
                'cx': sum(xs) / 4,
                'cy': sum(ys) / 4,
                'left': min(xs),
                'right': max(xs),
                'top': min(ys),
                'bottom': max(ys),
                'width': max(xs) - min(xs),
                'height': max(ys) - min(ys),
            })

        # Circled number to field mapping (main registration section)
        # Structure: marker → label → value (in sequence to the right)
        marker_field_map = {
            '①': ('vehicle_no', '자동차등록번호'),
            '②': ('vehicle_type', '차종'),
            '③': ('model_name', '차명'),
            '④': ('vehicle_format', '형식'),
            '⑤': ('model_year', '모델연도'),
            '⑥': ('vin', '차대번호'),
            '⑦': ('engine_type', '원동기형식'),
            '⑧': ('owner_name', '성명'),
            '⑨': ('owner_id', '주민등록번호'),
            '⑩': ('owner_address', '사용본거지'),
        }

        # Labels to skip when looking for values
        skip_labels = {'자동차등록번호', '차종', '차명', '형식', '모델연도', '연식',
                       '차대번호', '원동기형식', '성명', '명칭', '사용본거지',
                       '차', '종', '명', '주민', '법인', '등록번호', '형식 및 연식'}

        # Sort boxes by position for sequential lookup
        sorted_boxes = sorted(indexed_boxes, key=lambda b: (b['cy'], b['cx']))

        # Find markers in upper part of document (main registration section, y < 40%)
        main_section_limit = image_height * 0.45

        for i, box in enumerate(sorted_boxes):
            if box['cy'] > main_section_limit:
                continue  # Skip spec section

            text = box['text']

            for marker, (field_name, korean_label) in marker_field_map.items():
                if marker not in text:
                    continue
                if field_name in result:
                    continue  # Already found

                # Found a marker - look for value in subsequent boxes on same row
                row_tolerance = 60  # pixels
                candidates = []

                for j in range(i + 1, min(i + 10, len(sorted_boxes))):
                    other = sorted_boxes[j]

                    # Must be on same row (similar Y)
                    if abs(other['cy'] - box['cy']) > row_tolerance:
                        continue

                    # Must be to the right
                    if other['left'] < box['right'] - 30:
                        continue

                    # Skip other markers
                    if any(m in other['text'] for m in marker_field_map.keys()):
                        continue

                    # Skip labels
                    other_clean = other['text'].strip()
                    if other_clean in skip_labels or len(other_clean) <= 1:
                        continue

                    # This could be a value
                    candidates.append(other)

                # Get the best candidate (skip first if it's a label, take second)
                for candidate in candidates:
                    value = self._clean_field_value(candidate['text'], field_name, korean_label)
                    if value and len(value) >= 2:
                        # Additional validation for specific fields
                        if field_name == 'vin' and len(value) != 17:
                            continue
                        if field_name == 'model_year' and not re.match(r'^(19|20)\d{2}$', value):
                            continue
                        result[field_name] = value
                        confidences[field_name] = candidate['score']
                        break

        # Special handling for VIN - look for 17-char alphanumeric
        # OCR often inserts punctuation (. , ') and confuses L/I/1
        if 'vin' not in result:
            for box in indexed_boxes:
                if box['cy'] > main_section_limit:
                    continue

                text_upper = box['text'].upper()

                # First try exact 17-char match
                vin_match = re.search(r'\b([A-Z0-9]{17})\b', text_upper)
                if vin_match:
                    result['vin'] = vin_match.group(1)
                    confidences['vin'] = box['score']
                    break

                # Try with punctuation removal for Korean VIN prefixes
                # Pattern: [L/I/1][./,]?K[L/I/1]?[./,]?A... (handles LKLA, L.KLA, IKL.A, etc.)
                clean_text = re.sub(r'[^A-Z0-9]', '', text_upper)
                for prefix in ['LK', 'LA', 'KM', 'KN', 'KL', 'LC', 'LD']:
                    idx = clean_text.find(prefix)
                    if idx >= 0 and len(clean_text) >= idx + 17:
                        candidate = clean_text[idx:idx+17]
                        if len(candidate) == 17 and candidate.isalnum():
                            # Fix common OCR errors
                            if candidate[0] in '1I':
                                candidate = 'L' + candidate[1:]
                            result['vin'] = candidate
                            confidences['vin'] = box['score'] * 0.9  # Slightly lower confidence
                            break

                    # Also check for OCR error at first char (1KLA, IKLA -> LKLA)
                    for wrong, correct in [('1K', 'LK'), ('IK', 'LK'), ('1A', 'LA'), ('1C', 'LC')]:
                        idx = clean_text.find(wrong)
                        if idx >= 0 and len(clean_text) >= idx + 17:
                            candidate = correct + clean_text[idx+2:idx+17]
                            if len(candidate) == 17 and candidate.isalnum():
                                result['vin'] = candidate
                                confidences['vin'] = box['score'] * 0.85
                                break

                if 'vin' in result:
                    break

        # Special handling for vehicle_type - look for "승용", "승합" etc
        if 'vehicle_type' not in result:
            for box in indexed_boxes:
                if box['cy'] > main_section_limit:
                    continue
                type_match = re.search(r'(대형|중형|소형|경형)\s*(승용|승합|화물|특수)', box['text'])
                if type_match:
                    result['vehicle_type'] = type_match.group(0).replace(' ', '')
                    confidences['vehicle_type'] = box['score']
                    break

        # Special handling for vehicle_no - look for Korean plate pattern
        if 'vehicle_no' not in result or not self._is_valid_vehicle_no(result.get('vehicle_no', '')):
            regions = '서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주'
            usage = '자아바사허하거너더러머버서어저고노도로모보소오조구누두루무부수우주가나다라마'
            pattern = rf'({regions})\s*(\d{{2,3}})\s*([{usage}])\s*(\d{{4}})'

            for box in indexed_boxes:
                if box['cy'] > main_section_limit:
                    continue
                match = re.search(pattern, box['text'])
                if match:
                    result['vehicle_no'] = f"{match.group(1)}{match.group(2)}{match.group(3)}{match.group(4)}"
                    confidences['vehicle_no'] = box['score']
                    break

        # Special handling for model_year - look for 4-digit year
        if 'model_year' not in result:
            for box in indexed_boxes:
                if box['cy'] > main_section_limit:
                    continue
                # Look for year after "모델연도" or standalone
                if '모델연도' in box['text'] or '연식' in box['text']:
                    year_match = re.search(r'(20\d{2})', box['text'])
                    if year_match:
                        result['model_year'] = year_match.group(1)
                        confidences['model_year'] = box['score']
                        break
            # If still not found, look for standalone year
            if 'model_year' not in result:
                for box in indexed_boxes:
                    if box['cy'] > main_section_limit:
                        continue
                    if re.match(r'^20\d{2}$', box['text'].strip()):
                        result['model_year'] = box['text'].strip()
                        confidences['model_year'] = box['score']
                        break

        # Special handling for owner_name - look for Korean name pattern
        if 'owner_name' not in result or result.get('owner_name', '').startswith('주민'):
            for box in indexed_boxes:
                if box['cy'] > main_section_limit:
                    continue
                # Korean name: 2-4 Korean characters, not a label
                text = box['text'].strip()
                if (re.match(r'^[가-힣]{2,4}$', text) and
                    text not in skip_labels and
                    '주민' not in text and '법인' not in text):
                    result['owner_name'] = text
                    confidences['owner_name'] = box['score']
                    break

        # Special handling for engine_type
        if 'engine_type' not in result:
            for box in indexed_boxes:
                if box['cy'] > main_section_limit:
                    continue
                # Engine type often starts with letter+numbers or EM/TZ etc
                engine_match = re.search(r'\b([A-Z]{2,3}\d{2,4}[A-Z0-9\-]*)\b', box['text'].upper())
                if engine_match:
                    result['engine_type'] = engine_match.group(1)
                    confidences['engine_type'] = box['score']
                    break

        # Spec section marker-based extraction (⑪-㉒)
        spec_marker_map = {
            '⑪': ('spec_no', '제원관리번호'),
            '⑫': ('length_mm', '길이'),
            '⑬': ('width_mm', '너비'),
            '⑭': ('height_mm', '높이'),
            '⑮': ('total_weight_kg', '총중량'),
            '⑯': ('displacement_cc', '배기량'),
            '⑰': ('rated_output', '정격출력'),
            '⑱': ('passenger_capacity', '승차정원'),
            '⑲': ('max_load_kg', '최대적재량'),
            '⑳': ('cylinders', '기통수'),
            '㉑': ('type_approval_no', '형식승인번호'),
            '㉒': ('fuel_type', '연료의종류'),
        }

        # Find spec section markers (lower part of document)
        spec_section_start = image_height * 0.40

        for box in indexed_boxes:
            if box['cy'] < spec_section_start:
                continue  # Skip main section

            text = box['text']

            for marker, (field_name, korean_label) in spec_marker_map.items():
                if marker not in text:
                    continue
                if field_name in result:
                    continue  # Already found

                # Extract value after marker or label
                value_text = text.replace(marker, '').replace(korean_label, '').strip()

                # Handle fuel type
                if field_name == 'fuel_type':
                    for korean, english in self.FUEL_TYPES.items():
                        if korean in text:
                            result[field_name] = english
                            confidences[field_name] = box['score']
                            break
                # Handle numeric fields
                elif value_text:
                    num_match = re.search(r'(\d+)', value_text)
                    if num_match:
                        result[field_name] = num_match.group(1)
                        confidences[field_name] = box['score']

        # Extract spec section fields using label detection (fallback)
        spec_labels = {
            '길이': 'length_mm',
            '너비': 'width_mm',
            '높이': 'height_mm',
            '총중량': 'total_weight_kg',
            '배기량': 'displacement_cc',
            '승차정원': 'passenger_capacity',
            '연료의종류': 'fuel_type',
            '연료': 'fuel_type',
        }

        for box in indexed_boxes:
            for label, field_name in spec_labels.items():
                if label in box['text'] and field_name not in result:
                    text = box['text'].replace(label, '').strip()
                    num_match = re.search(r'(\d+)', text)
                    if num_match:
                        result[field_name] = num_match.group(1)
                        confidences[field_name] = box['score']
                    elif label == '연료' or label == '연료의종류':
                        for korean, english in self.FUEL_TYPES.items():
                            if korean in box['text']:
                                result[field_name] = english
                                confidences[field_name] = box['score']
                                break

        result = self._post_process(result)
        result['confidences'] = confidences

        return result

    def _is_valid_vehicle_no(self, vehicle_no: str) -> bool:
        """Check if vehicle_no looks like a valid Korean plate number."""
        if not vehicle_no:
            return False
        regions = '서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주'
        usage = '자아바사허하거너더러머버서어저고노도로모보소오조구누두루무부수우주가나다라마'
        pattern = rf'^({regions})\d{{2,3}}[{usage}]\d{{4}}$'
        return bool(re.match(pattern, vehicle_no))

    def _clean_field_value(self, raw_text: str, field_name: str, korean_label: str) -> Optional[str]:
        """Clean and extract value from raw OCR text."""
        if not raw_text:
            return None

        # Remove the field label itself
        value = raw_text.replace(korean_label, '').strip()

        # Remove circled numbers
        circled = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒'
        for c in circled:
            value = value.replace(c, '')

        # Remove common label words
        labels = ['자동차등록번호', '차종', '차명', '형식', '모델연도', '연식',
                  '차대번호', '원동기형식', '성명', '명칭', '사용본거지',
                  '제원관리번호', '길이', '너비', '높이', '총중량', '배기량',
                  '정격출력', '승차정원', '최대적재량', '연료의종류', '용도']
        for label in labels:
            value = value.replace(label, '')

        value = re.sub(r'^[\s:：]+', '', value).strip()
        value = re.sub(r'[\s:：]+$', '', value).strip()

        # Field-specific cleaning
        if field_name == 'vin':
            # Extract 17-char VIN
            vin_match = re.search(r'[A-Z0-9]{17}', value.upper())
            if vin_match:
                return vin_match.group(0)

        elif field_name == 'vehicle_no':
            # Extract vehicle number pattern
            regions = '서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주'
            usage = '자아바사허하거너더러머버서어저고노도로모보소오조구누두루무부수우주가나다라마'
            pattern = rf'({regions})\s*(\d{{2,3}})\s*([{usage}])\s*(\d{{4}})'
            match = re.search(pattern, value)
            if match:
                return f"{match.group(1)}{match.group(2)}{match.group(3)}{match.group(4)}"

        elif field_name == 'model_year':
            # Extract 4-digit year
            year_match = re.search(r'(19|20)\d{2}', value)
            if year_match:
                return year_match.group(0)

        elif field_name in ['length_mm', 'width_mm', 'height_mm']:
            # Extract number
            num_match = re.search(r'(\d{3,5})', value)
            if num_match:
                return num_match.group(1)

        elif field_name in ['passenger_capacity']:
            # Extract passenger count
            num_match = re.search(r'(\d{1,3})', value)
            if num_match:
                return num_match.group(1)

        elif field_name == 'fuel_type':
            # Extract fuel type
            for korean, english in self.FUEL_TYPES.items():
                if korean in value:
                    return english

        return value if value else None

    def _post_process(self, result: Dict) -> Dict:
        """Post-process extracted fields."""
        # Normalize fuel type
        if result.get('fuel_type') and result['fuel_type'] not in self.FUEL_TYPES.values():
            for korean, english in self.FUEL_TYPES.items():
                if korean in result['fuel_type']:
                    result['fuel_type'] = english
                    break

        # Clean vehicle_no
        if result.get('vehicle_no'):
            result['vehicle_no'] = re.sub(r'\s+', '', result['vehicle_no'])

        return result

    def learn_template_from_samples(self, sample_results: List[Dict]) -> Dict:
        """
        Learn/refine template from multiple sample OCR results.

        Args:
            sample_results: List of OCR results with boxes and known field values

        Returns:
            Updated template configuration
        """
        # This method would analyze multiple samples to refine field regions
        # For now, return current template
        template = {
            'main_fields': [(f.name, f.x1, f.y1, f.x2, f.y2) for f in self.MAIN_FIELDS],
            'spec_fields': [(f.name, f.x1, f.y1, f.x2, f.y2) for f in self.SPEC_FIELDS],
        }
        return template
