# -*- coding: utf-8 -*-
"""
Structured parser for Korean Vehicle Registration Certificate (자동차등록증).
Uses position-based extraction focusing on circled numbers ①-㉒.
"""
import re
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class StructuredRegistrationParser:
    """
    Position-based parser for vehicle registration certificates.
    Focuses on circled number markers (①-㉒) and their corresponding values.
    """

    # Unicode for circled numbers ① to ㉒
    CIRCLED_NUMBERS = {
        '\u2460': 1,   # ①
        '\u2461': 2,   # ②
        '\u2462': 3,   # ③
        '\u2463': 4,   # ④
        '\u2464': 5,   # ⑤
        '\u2465': 6,   # ⑥
        '\u2466': 7,   # ⑦
        '\u2467': 8,   # ⑧
        '\u2468': 9,   # ⑨
        '\u2469': 10,  # ⑩
        '\u246a': 11,  # ⑪
        '\u246b': 12,  # ⑫
        '\u246c': 13,  # ⑬
        '\u246d': 14,  # ⑭
        '\u246e': 15,  # ⑮
        '\u246f': 16,  # ⑯
        '\u2470': 17,  # ⑰
        '\u2471': 18,  # ⑱
        '\u2472': 19,  # ⑲
        '\u2473': 20,  # ⑳
        '\u3251': 21,  # ㉑
        '\u3252': 22,  # ㉒
    }

    # Reverse mapping: number to unicode
    NUMBER_TO_CIRCLED = {v: k for k, v in CIRCLED_NUMBERS.items()}

    # Main registration info fields (top section)
    # These circled numbers appear in the main registration area
    MAIN_FIELDS = {
        1: ('vehicle_no', '자동차등록번호'),
        2: ('vehicle_type', '차종'),
        3: ('model_name', '차명'),
        4: ('vehicle_format', '형식'),
        5: ('model_year', '모델연도'),
        6: ('vin', '차대번호'),
        7: ('engine_type', '원동기형식'),
        8: ('owner_name', '성명'),
        9: ('owner_id', '주민등록번호'),  # Usually redacted
        10: ('owner_address', '사용본거지'),
    }

    # Specification fields (제원 section)
    # Circled numbers ⑪-㉒ (11-22) in the spec section
    SPEC_FIELDS = {
        11: ('spec_management_no', '제원관리번호'),
        12: ('length', '길이'),
        13: ('width', '너비'),
        14: ('height', '높이'),
        15: ('total_weight', '총중량'),
        16: ('displacement', '배기량'),
        17: ('rated_output', '정격출력'),
        18: ('passenger_capacity', '승차정원'),
        19: ('max_load', '최대적재량'),
        20: ('cylinders', '기통수'),
        21: ('type_approval_no', '형식승인번호'),
        22: ('fuel_type', '연료의종류'),
    }

    # Fuel type mappings
    FUEL_TYPES = {
        '전기': 'Electric',
        'CNG': 'CNG',
        '천연가스': 'CNG',
        '디젤': 'Diesel',
        '경유': 'Diesel',
        'LPG': 'LPG',
        '엘피지': 'LPG',
        '수소': 'Hydrogen',
        '하이브리드': 'Hybrid',
        '휘발유': 'Gasoline',
        '가솔린': 'Gasoline',
    }

    # VIN prefixes for Korean vehicles
    VIN_PREFIXES = ('KM', 'KL', 'KN', 'KP', 'LA', 'LF', 'LG', 'LJ')

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_with_boxes(self, ocr_result: List[Tuple]) -> Dict[str, Any]:
        """
        Parse OCR result with bounding boxes using position-based extraction.

        Args:
            ocr_result: List of (box, (text, confidence)) tuples from PaddleOCR
                       box format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns:
            Dictionary with extracted fields and confidences
        """
        if not ocr_result:
            return {}

        # Build spatial index of all text elements
        elements = self._build_element_index(ocr_result)

        # Separate into sections based on Y position
        main_section, spec_section = self._separate_sections(elements)

        # Extract fields from each section
        result = {}
        confidences = {}

        # Extract main registration fields
        main_data = self._extract_section_fields(main_section, self.MAIN_FIELDS)
        result.update(main_data['fields'])
        confidences.update(main_data['confidences'])

        # Extract specification fields
        spec_data = self._extract_section_fields(spec_section, self.SPEC_FIELDS)
        # Prefix spec fields to avoid collision
        for key, value in spec_data['fields'].items():
            if key not in result:  # Don't override main fields
                result[key] = value
                confidences[key] = spec_data['confidences'].get(key, 0.0)

        # Post-process specific fields
        result = self._post_process(result)
        result['confidences'] = confidences

        return result

    def _build_element_index(self, ocr_result: List[Tuple]) -> List[Dict]:
        """
        Build a list of text elements with their positions.

        Returns list of dicts: {
            'text': str,
            'confidence': float,
            'box': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
            'center_x': float,
            'center_y': float,
            'left': float,
            'right': float,
            'top': float,
            'bottom': float,
            'circled_num': int or None
        }
        """
        elements = []

        for item in ocr_result:
            if not item or len(item) < 2:
                continue

            box = item[0]
            text_data = item[1]

            if isinstance(text_data, tuple) and len(text_data) >= 2:
                text = text_data[0]
                confidence = text_data[1]
            else:
                text = str(text_data)
                confidence = 0.0

            if not text or not text.strip():
                continue

            # Calculate bounding box metrics
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]

            element = {
                'text': text.strip(),
                'confidence': confidence,
                'box': box,
                'center_x': sum(xs) / 4,
                'center_y': sum(ys) / 4,
                'left': min(xs),
                'right': max(xs),
                'top': min(ys),
                'bottom': max(ys),
                'circled_num': self._extract_circled_number(text),
            }
            elements.append(element)

        return elements

    def _extract_circled_number(self, text: str) -> Optional[int]:
        """Extract circled number from text if present."""
        for char in text:
            if char in self.CIRCLED_NUMBERS:
                return self.CIRCLED_NUMBERS[char]
        return None

    def _separate_sections(self, elements: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Separate elements into main registration section and spec section.

        The spec section typically starts after "1. 제원" label and is in the lower
        portion of the document.
        """
        if not elements:
            return [], []

        # Find "제원" marker to identify spec section
        spec_marker_y = None
        for elem in elements:
            if '제원' in elem['text'] or '1.' in elem['text']:
                # Check if this is the "1. 제원" section header
                if any('제원' in e['text'] for e in elements
                       if abs(e['center_y'] - elem['center_y']) < 30):
                    spec_marker_y = elem['top']
                    break

        if spec_marker_y is None:
            # Fallback: use Y position threshold (spec section is typically in lower 60%)
            all_ys = [e['center_y'] for e in elements]
            if all_ys:
                y_range = max(all_ys) - min(all_ys)
                spec_marker_y = min(all_ys) + y_range * 0.4

        main_section = []
        spec_section = []

        for elem in elements:
            if spec_marker_y and elem['center_y'] > spec_marker_y:
                spec_section.append(elem)
            else:
                main_section.append(elem)

        return main_section, spec_section

    def _extract_section_fields(
        self,
        elements: List[Dict],
        field_mapping: Dict[int, Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Extract fields from a section using circled number positions.

        Args:
            elements: List of text elements in the section
            field_mapping: Dict mapping circled number to (field_name, korean_label)

        Returns:
            Dict with 'fields' and 'confidences'
        """
        fields = {}
        confidences = {}

        # Find all circled number markers
        markers = [e for e in elements if e['circled_num'] is not None]

        for marker in markers:
            num = marker['circled_num']
            if num not in field_mapping:
                continue

            field_name, korean_label = field_mapping[num]

            # Find the corresponding value
            value, confidence = self._find_value_for_marker(
                marker, elements, korean_label
            )

            if value:
                fields[field_name] = value
                confidences[field_name] = confidence

        return {'fields': fields, 'confidences': confidences}

    def _find_value_for_marker(
        self,
        marker: Dict,
        elements: List[Dict],
        korean_label: str
    ) -> Tuple[Optional[str], float]:
        """
        Find the value corresponding to a circled number marker.

        Strategy:
        1. Check if marker text itself contains value after the label
        2. Look for text to the right of the marker (same row)
        3. Look for text below the marker (if label is above value)
        4. Exclude other markers and known labels
        """
        candidates = []

        marker_right = marker['right']
        marker_center_y = marker['center_y']
        marker_bottom = marker['bottom']

        # Common labels to exclude (these are field names, not values)
        label_keywords = [
            '자동차등록번호', '차종', '차명', '형식', '모델연도', '연식',
            '차대번호', '원동기형식', '성명', '명칭', '주민등록번호', '법인등록번호',
            '사용본거지', '제원관리번호', '길이', '너비', '높이', '총중량',
            '배기량', '정격출력', '승차정원', '최대적재량', '기통수', '연료의종류',
            '형식승인번호', '용도', '주', '번호', '등록', '형식승인'
        ]

        # Strategy 0: Check if marker text itself contains value
        # e.g., "① 제주79자7052" -> extract "제주79자7052"
        marker_text = marker['text']
        # Remove circled numbers and labels from marker text
        potential_value = marker_text
        for char in self.CIRCLED_NUMBERS.keys():
            potential_value = potential_value.replace(char, '')
        potential_value = potential_value.strip()

        # Check if remaining text is a value (not just a label)
        is_label_only = any(label in potential_value for label in label_keywords)
        if potential_value and not is_label_only and len(potential_value) > 1:
            # Check if it looks like actual data (contains numbers or is long enough)
            has_numbers = any(c.isdigit() for c in potential_value)
            is_long_enough = len(potential_value) >= 4
            if has_numbers or is_long_enough:
                return potential_value, marker['confidence']

        for elem in elements:
            # Skip the marker itself
            if elem is marker:
                continue

            # Skip elements containing only circled numbers
            if elem['circled_num'] is not None:
                elem_text_clean = elem['text']
                for char in self.CIRCLED_NUMBERS.keys():
                    elem_text_clean = elem_text_clean.replace(char, '')
                if not elem_text_clean.strip():
                    continue

            # Skip if text contains only label keywords
            text_clean = elem['text'].strip()
            is_only_label = any(text_clean == label or text_clean.endswith(label)
                               for label in label_keywords[:6])  # Major labels
            if is_only_label and len(text_clean) < 10:
                continue

            # Skip very short text (likely noise)
            if len(text_clean) < 2:
                continue

            # Strategy 1: To the right, same row (within Y tolerance)
            y_tolerance = (marker['bottom'] - marker['top']) * 2.0
            if (elem['left'] > marker_right - 20 and
                abs(elem['center_y'] - marker_center_y) < y_tolerance):
                # Calculate score based on proximity
                x_distance = max(0, elem['left'] - marker_right)
                score = 1.0 / (1.0 + x_distance / 100)

                # Boost score if text looks like actual data
                if any(c.isdigit() for c in elem['text']):
                    score *= 1.5
                if len(elem['text']) > 5:
                    score *= 1.2

                candidates.append((elem, score, 'right'))

            # Strategy 2: Below the marker (for stacked layouts)
            x_tolerance = (marker['right'] - marker['left']) * 3
            if (elem['top'] > marker_bottom - 5 and
                elem['top'] < marker_bottom + 80 and
                abs(elem['center_x'] - marker['center_x']) < x_tolerance):
                y_distance = elem['top'] - marker_bottom
                score = 0.7 / (1.0 + y_distance / 50)

                # Boost score if text looks like actual data
                if any(c.isdigit() for c in elem['text']):
                    score *= 1.5

                candidates.append((elem, score, 'below'))

        if not candidates:
            return None, 0.0

        # Sort by score and get best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_elem, _, _ = candidates[0]

        # Clean the value
        value = self._clean_value(best_elem['text'], korean_label)

        # Final validation - reject if it's still just a label
        if value in label_keywords:
            return None, 0.0

        return value, best_elem['confidence']

    def _clean_value(self, text: str, field_label: str) -> str:
        """Clean extracted value by removing labels and extra characters."""
        value = text.strip()

        # Remove the field label if it's included
        value = value.replace(field_label, '').strip()

        # Remove common prefixes/suffixes
        value = re.sub(r'^[:\s]+', '', value)
        value = re.sub(r'[:\s]+$', '', value)

        # Remove circled numbers that might have been captured
        for char in self.CIRCLED_NUMBERS.keys():
            value = value.replace(char, '')

        return value.strip()

    def _post_process(self, result: Dict) -> Dict:
        """Post-process extracted fields for validation and normalization."""

        # Normalize vehicle number
        if result.get('vehicle_no'):
            result['vehicle_no'] = self._normalize_vehicle_no(result['vehicle_no'])

        # Validate and clean VIN
        if result.get('vin'):
            vin = self._clean_vin(result['vin'])
            if self._is_valid_vin(vin):
                result['vin'] = vin
            else:
                result['vin'] = None

        # Normalize fuel type
        if result.get('fuel_type'):
            result['fuel_type'] = self._normalize_fuel_type(result['fuel_type'])

        # Clean model year
        if result.get('model_year'):
            year_match = re.search(r'(19|20)\d{2}', str(result['model_year']))
            if year_match:
                result['model_year'] = year_match.group(0)

        # Clean registration date
        if result.get('registration_date'):
            result['registration_date'] = self._normalize_date(result['registration_date'])

        return result

    def _normalize_vehicle_no(self, vehicle_no: str) -> Optional[str]:
        """Normalize Korean vehicle registration number."""
        if not vehicle_no:
            return None

        # Remove whitespace
        normalized = re.sub(r'\s+', '', vehicle_no)

        # Valid patterns
        regions = '서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주'
        usage = '자아바사허하거너더러머버서어저고노도로모보소오조구누두루무부수우주가나다라마'

        # Pattern: Region + 2-3 digits + usage char + 4 digits
        pattern = rf'^({regions})\d{{2,3}}[{usage}]\d{{4}}$'
        if re.match(pattern, normalized):
            return normalized

        # Pattern: 3 digits + Korean + 4 digits (new format)
        pattern2 = r'^\d{3}[가-힣]\d{4}$'
        if re.match(pattern2, normalized):
            return normalized

        return normalized if len(normalized) >= 7 else None

    def _clean_vin(self, vin: str) -> str:
        """
        Clean VIN by removing invalid characters and normalizing.
        VIN cannot contain I, O, Q (they look like 1, 0, 0).
        Korean vehicle VINs typically start with: LK, LA, KM, KN, KL, LC, LD, etc.
        """
        if not vin:
            return ''

        # Remove whitespace and convert to uppercase
        vin = re.sub(r'\s+', '', vin).upper()

        # Remove ALL punctuation first (OCR often inserts . , ' etc.)
        vin = re.sub(r'[^A-Z0-9]', '', vin)

        if len(vin) < 17:
            return vin

        # Take first 17 characters
        vin_chars = list(vin[:17])

        # Step 1: Replace I, O, Q everywhere first (these are NEVER valid in VIN)
        for i in range(len(vin_chars)):
            if vin_chars[i] == 'O':
                vin_chars[i] = '0'
            elif vin_chars[i] == 'Q':
                vin_chars[i] = '0'
            elif vin_chars[i] == 'I':
                vin_chars[i] = '1'  # Default: I -> 1

        # Step 2: Smart correction for Korean VIN prefixes
        # Position 0 (WMI Country): L or K for Korean vehicles
        if vin_chars[0] == '1':
            # Check if this looks like a Korean VIN pattern
            if vin_chars[1] in 'AKLN' or (vin_chars[1] == '1' and vin_chars[2] in 'AL'):
                vin_chars[0] = 'L'

        # Position 1: For LK, LA, KM, KN, KL patterns
        if vin_chars[1] == '1':
            if vin_chars[0] == 'L':
                vin_chars[1] = 'K'  # L1 -> LK (common Korean prefix)
            elif vin_chars[0] == 'K':
                vin_chars[1] = 'L'  # K1 -> KL

        # Position 2: Often L or A
        if vin_chars[2] == '1':
            if vin_chars[0:2] == ['L', 'K']:
                vin_chars[2] = 'L'  # LK1 -> LKL (for LKLA pattern)

        # Position 5: Often E in Korean VINs (e.g., LKLA6E1X...)
        # Pattern: LKLA6X1X -> likely LKLA6E1X
        if vin_chars[5] == '1' and vin_chars[0:4] == ['L', 'K', 'L', 'A']:
            if vin_chars[4].isdigit() and vin_chars[7] in 'X0123456789':
                vin_chars[5] = 'E'  # LKLA61 -> LKLA6E

        return ''.join(vin_chars)

    def _extract_vin_aggressive(self, text: str) -> Optional[str]:
        """
        Aggressively extract VIN from text with OCR error correction.
        Used as fallback when standard patterns fail.
        """
        if not text:
            return None

        # Clean all text first - remove ALL non-alphanumeric
        clean_text = re.sub(r'[^A-Za-z0-9]', '', text).upper()

        # Look for Korean VIN prefixes and try to extract 17 chars
        prefixes = ['LK', 'LA', 'KM', 'KN', 'KL', 'KP', 'LC', 'LD', 'LF', 'LG', 'LJ']

        for prefix in prefixes:
            # Find the prefix in text
            idx = clean_text.find(prefix)
            if idx >= 0:
                # Try to extract 17 characters starting from this prefix
                candidate = clean_text[idx:idx+17]
                if len(candidate) == 17:
                    # Apply cleaning
                    cleaned = self._clean_vin(candidate)
                    if self._is_valid_vin(cleaned):
                        return cleaned

        # Also try looking for patterns like 1KLA, IKL.A that might be LKLA
        # Fix common first-character OCR errors
        for wrong_prefix, correct_prefix in [('1K', 'LK'), ('IK', 'LK'), ('1A', 'LA'),
                                              ('1C', 'LC'), ('1D', 'LD')]:
            idx = clean_text.find(wrong_prefix)
            if idx >= 0:
                candidate = correct_prefix + clean_text[idx+2:idx+17]
                if len(candidate) == 17:
                    cleaned = self._clean_vin(candidate)
                    if self._is_valid_vin(cleaned):
                        return cleaned

        return None

    def _is_valid_vin(self, vin: str) -> bool:
        """Validate VIN format."""
        if not vin or len(vin) != 17:
            return False

        # Should not contain I, O, Q
        if any(c in vin for c in 'IOQ'):
            return False

        return vin.isalnum()

    def _normalize_fuel_type(self, fuel_type: str) -> str:
        """Normalize fuel type to standardized English."""
        if not fuel_type:
            return 'Unknown'

        fuel_type_clean = fuel_type.strip()

        for korean, english in self.FUEL_TYPES.items():
            if korean in fuel_type_clean:
                return english

        return fuel_type_clean if fuel_type_clean else 'Unknown'

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to YYYY-MM-DD format."""
        if not date_str:
            return ''

        # Remove Korean date suffixes
        normalized = re.sub(r'[년월일\s]+', '-', date_str)
        normalized = re.sub(r'-+', '-', normalized).strip('-')

        return normalized

    def verify_document_type(self, text: str) -> bool:
        """Verify if text contains vehicle registration certificate keywords."""
        if not text:
            return False

        clean_text = re.sub(r'\s+', '', text)

        keywords = ['자동차등록증', '차대번호', '자동차등록번호']
        for keyword in keywords:
            if keyword in clean_text:
                return True

        return False

    def parse_from_paddle_result(self, paddle_result: Dict) -> Dict[str, Any]:
        """
        Parse from PaddleOCR result format.

        Args:
            paddle_result: Dict with 'result' key containing raw OCR boxes

        Returns:
            Parsed fields dictionary
        """
        raw_result = paddle_result.get('result', [])
        return self.parse_with_boxes(raw_result)

    def extract_by_circled_patterns(self, full_text: str) -> Dict[str, Any]:
        """
        Extract all 22 fields using regex patterns based on circled number markers ①-㉒.

        Field mapping:
        === Main Registration (상단 등록정보) ===
        ① 자동차등록번호 (vehicle_no)
        ② 차종 (vehicle_type)
        ③ 차명 (model_name)
        ④ 형식 (vehicle_format)
        ⑤ 모델연도 (model_year)
        ⑥ 차대번호 (vin)
        ⑦ 원동기형식 (engine_type)
        ⑧ 성명/명칭 (owner_name)
        ⑨ 주민/법인등록번호 (owner_id) - usually redacted
        ⑩ 사용본거지 (owner_address)

        === Specifications (제원) ===
        ① 제원관리번호 (spec_no)
        ② 길이 (length_mm)
        ③ 너비 (width_mm)
        ④ 높이 (height_mm)
        ⑤ 총중량 (total_weight_kg)
        ⑥ 배기량 (displacement_cc)
        ⑦ 정격출력 (rated_output)
        ⑧ 승차정원 (passenger_capacity)
        ⑨ 최대적재량 (max_load_kg)
        ⑩ 기통수 (cylinders)
        ⑪ 연료의종류 (fuel_type)
        ⑫ 형식승인번호 (type_approval_no)

        Args:
            full_text: Full OCR text joined with newlines

        Returns:
            Dictionary with extracted fields and confidences
        """
        result = {}
        confidences = {}

        # Clean text for pattern matching
        clean_text = re.sub(r'\s+', ' ', full_text)

        # Korean regions and usage characters
        regions = '서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주'
        usage = '자아바사허하거너더러머버서어저고노도로모보소오조구누두루무부수우주가나다라마'

        # ========================================
        # Main Registration Fields (①-⑩)
        # ========================================
        patterns = {
            # ① 자동차등록번호
            'vehicle_no': [
                rf'\u2460[^\n]*?((?:{regions})\s*\d{{2,3}}\s*[{usage}]\s*\d{{4}})',
                rf'자동차등록번호[^\n]*?((?:{regions})\s*\d{{2,3}}\s*[{usage}]\s*\d{{4}})',
                rf'\b((?:{regions})\s*\d{{2,3}}\s*[{usage}]\s*\d{{4}})\b',
                r'\b(\d{3}\s*[가-힣]\s*\d{4})\b',
            ],
            # ② 차종
            'vehicle_type': [
                r'\u2461[^①-㉒\n]*?([가-힣]+승[용합화])',
                r'차\s*종[^①-㉒\n]*?([가-힣]+(?:승용|승합|화물|특수))',
                r'\b((?:대형|중형|소형|경형|초소형)(?:승용|승합|화물|특수))\b',
            ],
            # ③ 차명
            'model_name': [
                r'차\s*명[^①-㉒\n]*?([A-Za-z][A-Za-z0-9\s\-()]{2,25})',
                r'\u2462[^①-㉒\n]*?([A-Za-z][A-Za-z0-9\s\-()]{2,25})',
                r'\b((?:ELEC\s*CITY|UNIVERSE|COUNTY|AERO|SUPER|HYPER|FIBIRD)[A-Za-z0-9\s\-]*)\b',
            ],
            # ④ 형식
            'vehicle_format': [
                r'형\s*식[^①-㉒\n]*?([A-Z0-9][A-Z0-9\-]{4,20})',
                r'\u2463[^①-㉒\n]*?([A-Z0-9][A-Z0-9\-]{4,20})',
            ],
            # ⑤ 모델연도
            'model_year': [
                r'\u2464[^①-㉒\n]*?((19|20)\d{2})',
                r'모델연도[^①-㉒\n]*?((19|20)\d{2})',
                r'연\s*식[^①-㉒\n]*?((19|20)\d{2})',
                r'\(모델연도\)\s*((19|20)\d{2})',
            ],
            # ⑥ 차대번호 (VIN)
            # Note: OCR often inserts punctuation (. , ') and confuses L/I/1
            'vin': [
                # Standard patterns (exact match)
                r'\u2465[^①-㉒\n]*?([A-Z0-9]{17})',
                r'차\s*대\s*번\s*호[^①-㉒\n]*?([A-Z0-9]{17})',
                # Prefix patterns (Korean VIN starts with LK, LA, KM, KN, etc.)
                r'\b(KM[A-HJ-NPR-Z0-9]{14})\b',
                r'\b(KN[A-HJ-NPR-Z0-9]{14})\b',
                r'\b(KL[A-HJ-NPR-Z0-9]{14})\b',
                r'\b(LK[A-HJ-NPR-Z0-9]{14})\b',
                r'\b(LA[A-HJ-NPR-Z0-9]{14})\b',
                r'\b(LC[A-HJ-NPR-Z0-9]{14})\b',
                r'\b(LD[A-HJ-NPR-Z0-9]{14})\b',
                # OCR error patterns - with punctuation (. , ') - needs capturing group
                r'([LI1][\.\,\']?K[LI1]?[\.\,\']?[A-Z][A-Z0-9\.\,\']{10,16})',
                r'([LI1][\.\,\']?A[A-Z0-9\.\,\']{13,17})',
                # OCR error - I or 1 instead of L at start
                r'([1I]K[A-Z][A-Z0-9]{13,15})',
            ],
            # ⑦ 원동기형식
            'engine_type': [
                r'원동기형식[^①-㉒\n]*?([A-Z0-9][A-Z0-9\-]{2,15})',
                r'\u2466[^①-㉒\n]*?([A-Z0-9][A-Z0-9\-]{2,15})',
            ],
            # ⑧ 성명(명칭)
            'owner_name': [
                r'성\s*명[^①-㉒\n]*?([가-힣]{2,10}(?:\s*\([가-힣]+\))?)',
                r'\u2467[^①-㉒\n]*?([가-힣]{2,10})',
                r'명\s*칭[^①-㉒\n]*?([가-힣()]{2,20})',
                r'소유자[^①-㉒\n]*?([가-힣()]{2,20})',
            ],
            # ⑩ 사용본거지
            'owner_address': [
                r'사용본거지[^①-㉒\n]*?([가-힣0-9\s\.\-,()]+(?:동|로|길|리|읍|면)[가-힣0-9\s\.\-,()]*)',
                r'\u2469[^①-㉒\n]*?([가-힣]+(?:시|도|구|군)[가-힣0-9\s\.\-,()]+)',
            ],

            # ========================================
            # Specification Fields (제원 ⑪-㉒)
            # ========================================
            # ⑪ 제원관리번호
            'spec_no': [
                r'제원관리번호[^①-㉒\n]*?([A-Z0-9\-]{5,20})',
            ],
            # ⑫ 길이
            'length_mm': [
                r'길\s*이[^①-㉒\n]*?(\d{4,5})\s*(?:mm|㎜)?',
            ],
            # ⑬ 너비
            'width_mm': [
                r'너\s*비[^①-㉒\n]*?(\d{4})\s*(?:mm|㎜)?',
            ],
            # ⑭ 높이
            'height_mm': [
                r'높\s*이[^①-㉒\n]*?(\d{4})\s*(?:mm|㎜)?',
            ],
            # ⑮ 총중량
            'total_weight_kg': [
                r'총\s*중\s*량[^①-㉒\n]*?(\d{3,5})\s*(?:kg|㎏)?',
            ],
            # ⑯ 배기량
            'displacement_cc': [
                r'배기량[^①-㉒\n]*?(\d{3,5})\s*(?:cc|CC|㏄)?',
            ],
            # ⑰ 정격출력
            'rated_output': [
                r'정격출력[^①-㉒\n]*?(\d{2,4}(?:[./]\d+)?)\s*(?:ps|kw|PS|KW|마력)?',
            ],
            # ⑱ 승차정원
            'passenger_capacity': [
                r'승차정원[^①-㉒\n]*?(\d{1,3})\s*인?',
            ],
            # ⑲ 최대적재량
            'max_load_kg': [
                r'최대적재량[^①-㉒\n]*?(\d{1,5})\s*(?:kg|㎏)?',
            ],
            # ⑳ 기통수
            'cylinders': [
                r'기통수[^①-㉒\n]*?(\d{1,2})',
            ],
            # ㉑ 형식승인번호
            'type_approval_no': [
                r'형식승인번호[^①-㉒\n]*?([A-Z0-9\-]{5,20})',
            ],
            # ㉒ 연료의종류
            'fuel_type': [
                r'㉒[^①-㉑\n]*?(전기|디젤|경유|휘발유|가솔린|LPG|CNG|수소|하이브리드)',
                r'연료[의]?\s*종류[^①-㉒\n]*?(전기|디젤|경유|휘발유|가솔린|LPG|CNG|수소|하이브리드)',
                r'사용연료[^①-㉒\n]*?(전기|디젤|경유|휘발유|가솔린|LPG|CNG|수소)',
            ],

            # ========================================
            # Additional Fields
            # ========================================
            # 최초등록일
            'first_registration_date': [
                r'최초등록일[^①-㉒\n]*?(\d{4}[\s\.\-년]*\d{1,2}[\s\.\-월]*\d{1,2})',
            ],
            # 용도
            'usage_type': [
                r'용\s*도[^①-㉒\n]*?(자가용|영업용|관용|개인택시|사업용)',
                r'\u2462용도[^①-㉒\n]*?(자가용|영업용|관용|개인택시)',
            ],
        }

        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, clean_text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    if value and len(value) >= 1:
                        result[field] = value
                        confidences[field] = 0.7
                        break

        # Post-processing
        if result.get('fuel_type'):
            result['fuel_type'] = self._normalize_fuel_type(result['fuel_type'])

        if result.get('first_registration_date'):
            result['first_registration_date'] = self._normalize_date(result['first_registration_date'])

        # Clean vehicle_no whitespace
        if result.get('vehicle_no'):
            result['vehicle_no'] = re.sub(r'\s+', '', result['vehicle_no'])

        # VIN post-processing - clean and validate
        if result.get('vin'):
            vin = self._clean_vin(result['vin'])
            if self._is_valid_vin(vin):
                result['vin'] = vin
            else:
                # Try aggressive extraction from the raw match
                aggressive_vin = self._extract_vin_aggressive(result['vin'])
                if aggressive_vin:
                    result['vin'] = aggressive_vin
                else:
                    result['vin'] = None

        # Fallback: If no VIN found, try aggressive extraction from full text
        if not result.get('vin'):
            aggressive_vin = self._extract_vin_aggressive(clean_text)
            if aggressive_vin:
                result['vin'] = aggressive_vin
                confidences['vin'] = 0.6  # Lower confidence for aggressive extraction

        result['confidences'] = confidences
        return result
