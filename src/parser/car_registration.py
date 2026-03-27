
# -*- coding: utf-8 -*-
import re
import logging


class CarRegistrationParser:
    """
    Parser for Korean Vehicle Registration Certificate (자동차등록증).
    Enhanced with multiple fallback patterns for better accuracy.
    """

    # Korean VIN prefixes (World Manufacturer Identifier)
    # KM: Hyundai, Kia
    # KL: GM Korea (Chevrolet)
    # KN: Renault Samsung
    # KP: SsangYong
    # LA: Chinese manufacturers (buses)
    VIN_PREFIXES = ('KM', 'KL', 'KN', 'KP', 'LA', 'LF', 'LG', 'LJ', 'KMJ', 'KMH', 'KNA')

    # Fuel type mappings (Korean -> English standardized)
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
        '바이오': 'Bio',
        '바이오디젤': 'BioDiesel',
    }

    # Keywords in OCR text that indicate fuel type
    FUEL_OCR_KEYWORDS = {
        'Electric': ['전기', '전기자동차', 'EV', 'ELEC', '일렉'],
        'CNG': ['CNG', '천연가스', '압축천연가스'],
        'Diesel': ['디젤', '경유'],
        'LPG': ['LPG', '엘피지', '액화석유가스'],
        'Hydrogen': ['수소', 'FCEV', '연료전지'],
        'Hybrid': ['하이브리드', 'HEV', 'PHEV'],
        'Gasoline': ['휘발유', '가솔린'],
    }

    def __init__(self):
        # Regex patterns using unicode escapes exclusively to avoid encoding issues
        self.fields = {
            # 1. Vehicle No (Automobile Registration Number)
            # Label: 자동차등록번호 (\uc790\ub3d9\ucc28\ub4f1\ub85d\ubc88\ud638)
            'vehicle_no': r'(?:\uc790\ub3d9\ucc28\ub4f1\ub85d\ubc88\ud638)[\s:]*([^\n]+?)(?=\s*(?:\u2461|\u2462|\ucc28\uc885)|\n|$)',

            # 2. VIN (Vehicle Identification Number) - Enhanced with multiple patterns
            # Label: 차대번호 (\ucc28\ub300\ubc88\ud638)
            # Pattern allows newline after label and handles OCR errors
            'vin': r'\ucc28\ub300\ubc88\ud638[^\n]*?[\s\n:]*([A-Z0-9]{17})',

            # [NEW] Vehicle Type (Type)
            # Label: 차종 (\ucc28\uc885)
            'vehicle_type': r'\ucc28\s*\uc885[\s:]*([^\n]+?)(?=\s*(?:\u2462|\u2463|\ucc28\uba85)|\n|$)',

            # 3. Model Name (Name)
            # Label: 차명 (\ucc28\uba85)
            'model_name': r'(?:\ucc28\s*\uba85|Model Name)[\s:]*([^\n\t/]+?)(?=\s*(?:\u2463|\u2464|\ud615\uc2dd)|/|\n|$)',

            # [NEW] Vehicle Format (Format)
            # Label: 형식 (\ud615\uc2dd)
            'vehicle_format': r'(?:\ud615\s*\uc2dd|\ud615\uc2dd\s*\ubc0f\s*\ubaa8\ub378\uc5f0\ub3c4)(?:.|\n){0,10}?\b([A-Z0-9-]{5,})(?=\s|/|\n)',

            # 4. Model Year (Model Year)
            # Label: 연식 (\uc5f0\uc2dd)
            'model_year': r'(?:\uc5f0\s*\uc2dd|\ubaa8\ub378\uc5f0\ub3c4)(?:.|\n){0,100}?\b((?:19|20)\d{2})\b',

            # 5. Engine Type (Engine Type)
            # Label: 원동기형식 (\uc6d0\ub3d9\uae30\ud615\uc2dd)
            'engine_type': r'\uc6d0\ub3d9\uae30\ud615\uc2dd(?:.|\n){0,50}?([A-Z0-9-]{5,})',

            # 6. Owner Name (Owner)
            # Label: 성명 (\uc131\uba85) | 소유자 (\uc18c\uc720\uc790)
            'owner_name': r'(?:\uc131\uba85|\uc18c\uc720\uc790)(?:[^\s:]*)?[\s:]*([^\n]+)',

            # 7. Address (User Base Address)
            # Label: 사용본거지 (\uc0ac\uc6a9\ubcf8\uac70\uc9c0)
            'owner_address': r'\uc0ac\uc6a9\ubcf8\uac70\uc9c0[\s:]*([^\n]+)',

            # 8. Registration Date (First Registration Date)
            # Label: 최초등록일 (\ucd5c\ucd08\ub4f1\ub85d\uc77c)
            'registration_date': r'\ucd5c\ucd08\ub4f1\ub85d\uc77c.*(\d{4}\s*[\uac00-\ud7a3-.]\s*\d{1,2}\s*[\uac00-\ud7a3-.]\s*\d{1,2})',

            # 9. Vehicle Specs (Specs Management No)
            # Label: 제원관리번호 (\uc81c\uc6d0\uad00\ub9ac\ubc88\ud638)
            'vehicle_specs': r'\uc81c\uc6d0\uad00\ub9ac\ubc88\ud638.*?([A-Z0-9-]+)',

            # 10. Length (길이) - value may be on next line
            'length_mm': r'\uae38\s*\uc774.{0,30}?(\d[\d,]{3,6})\s*(?:mm|\u339c)?',

            # 11. Width (너비) - value may be on next line
            'width_mm': r'\ub108\s*\ube44.{0,30}?(\d[\d,]{3,5})\s*(?:mm|\u339c)?',

            # 12. Height (높이) - value may be on next line
            'height_mm': r'\ub192\s*\uc774.{0,30}?(\d[\d,]{3,5})\s*(?:mm|\u339c)?',

            # 13. Total Weight (총중량) - value may be on next line
            'total_weight_kg': r'\ucd1d\s*\uc911?\s*\ub7c9.{0,30}?(\d[\d,]{3,6})\s*(?:kg|\u338f)?',

            # 14. Passenger Capacity (승차정원) - value may be on next line
            'passenger_capacity': r'\uc2b9\ucc28\uc815\uc6d0.{0,30}?(\d{1,3})\s*(?:\uba85|\uc778)?',
        }

        # Additional VIN patterns for fallback (ordered by priority)
        self.vin_patterns = [
            # Pattern 1: After "차대번호" label with newline support
            r'\ucc28\ub300\ubc88\ud638[^\n]*?[\s\n:]*([A-Z0-9]{17})',
            # Pattern 2: After ⑥ marker (circled 6)
            r'\u2465[^\n]*?[\s\n:]*([A-Z0-9]{17})',
            r'\u2466[^\n]*?[\s\n:]*([A-Z0-9]{17})',
            # Pattern 3: Direct VIN pattern (Korean vehicle prefixes)
            r'\b(KM[A-Z0-9]{14,15})\b',
            r'\b(KL[A-Z0-9]{14,15})\b',
            r'\b(KN[A-Z0-9]{14,15})\b',
            r'\b(LA[A-Z0-9]{14,15})\b',
            # Pattern 4: Any 17-char alphanumeric that looks like VIN
            r'\b([A-HJ-NPR-Z0-9]{17})\b',
        ]

        # Vehicle number patterns for fallback
        self.vehicle_no_patterns = [
            # Standard format: 지역 + 숫자 + 자/아/바 등 + 숫자
            r'(?:\uc790\ub3d9\ucc28\ub4f1\ub85d\ubc88\ud638)[\s:]*([^\n]+?)(?=\s*(?:\u2461|\u2462|\ucc28)|\n|$)',
            # Pattern with circled 1 (①)
            r'\u2460[^\n]*?[\s:]*([가-힣]+\s*\d+\s*[자아바사]?\s*\d+)',
            # Direct pattern: Korean region + number + 자 + number
            r'\b([가-힣]{2,4}\s*\d{2,3}\s*[자아바사]\s*\d{4})\b',
        ]

    def parse(self, text):
        if not text:
            return {}

        result = {}

        for field, pattern in self.fields.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                value = match.group(1).strip()
                # Cleaning
                if field == 'owner_name':
                    value = re.split(r'\s{2,}', value)[0]
                if field == 'registration_date':
                    value = re.sub(r'[\ub144\uc6d4\uc77c\s]+', '-', value).strip('-')

                # Check for bad captures (if value starts with bullet)
                bullets = ('\u2460', '\u2461', '\u2462', '\u2463', '\u2464', '\u2465', '\u2466')
                if value and value.startswith(bullets):
                    value = ''  # Reject bad capture

                result[field] = value
            else:
                result[field] = None

        # Enhanced VIN extraction with fallback patterns
        if not result.get('vin') or len(result.get('vin', '')) != 17:
            vin = self._extract_vin_fallback(text)
            if vin:
                result['vin'] = vin
                logging.info(f"VIN extracted via fallback: {vin}")

        # Enhanced Vehicle No extraction with fallback
        if not result.get('vehicle_no'):
            vehicle_no = self._extract_vehicle_no_enhanced(text)
            if vehicle_no:
                result['vehicle_no'] = vehicle_no
                logging.info(f"Vehicle No extracted via enhanced: {vehicle_no}")

        # Normalize vehicle number to standard format
        if result.get('vehicle_no'):
            normalized = self._normalize_vehicle_no(result['vehicle_no'])
            if normalized:
                result['vehicle_no'] = normalized

        return result

    def parse_single(self, ocr_text, filename=None):
        """
        Parse OCR text from a single engine and return complete result.
        Replaces parse_hybrid() for PaddleOCR-only mode.
        """
        result = self.parse(ocr_text) if ocr_text else {}

        # Fallback: extract vehicle_no from filename if OCR failed
        ocr_vehicle_no = result.get('vehicle_no')
        if filename and (not ocr_vehicle_no or not self._is_valid_vehicle_no(ocr_vehicle_no)):
            filename_vehicle_no = self._extract_vehicle_no_from_filename(filename)
            if filename_vehicle_no:
                logging.info(f"Vehicle No from filename: {filename_vehicle_no} (OCR was: '{ocr_vehicle_no}')")
                result['vehicle_no'] = filename_vehicle_no

        # Determine fuel type
        fuel_type = self._determine_fuel_type(
            filename=filename,
            ocr_text=ocr_text,
            model_name=result.get('model_name'),
            engine_type=result.get('engine_type'),
        )
        result['fuel_type'] = fuel_type

        return result

    def _extract_vin_fallback(self, text):
        """
        Extract VIN using multiple fallback patterns.
        Returns the first valid 17-character VIN found.
        """
        if not text:
            return None

        # Try each pattern in order
        for pattern in self.vin_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                vin = match.upper().strip()
                if self._is_valid_vin(vin):
                    return vin

        return None

    def _extract_vehicle_no_fallback(self, text):
        """
        Extract vehicle registration number using fallback patterns.
        """
        if not text:
            return None

        for pattern in self.vehicle_no_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                value = match.group(1).strip()
                # Clean up extra whitespace
                value = re.sub(r'\s+', ' ', value)
                if len(value) > 5:  # Minimum valid length
                    return value

        return None

    def _is_valid_vin(self, vin):
        """
        Validate VIN format.
        - Must be exactly 17 characters
        - Must not contain I, O, Q (commonly confused letters)
        - Should start with known prefixes for Korean vehicles
        """
        if not vin or len(vin) != 17:
            return False

        # VIN should not contain I, O, Q
        if any(c in vin for c in 'IOQ'):
            return False

        # Check if it's alphanumeric
        if not vin.isalnum():
            return False

        # Prefer VINs starting with known Korean prefixes
        # But accept others if they pass other checks
        return True

    def _normalize_vehicle_no(self, vehicle_no):
        """
        Normalize Korean vehicle registration number to standard format.

        Korean vehicle number formats:
        1. Commercial (bus/taxi): Region + 2digits + Usage(자/아/바/사) + 4digits
           Example: 강원70자1234, 서울70아5678

        2. New format: 3digits + Korean + 4digits
           Example: 123가4567

        3. Old format: Region + 2digits + Korean + 4digits
           Example: 서울12가3456

        Returns normalized format without spaces.
        """
        if not vehicle_no:
            return None

        # Remove all whitespace first
        normalized = re.sub(r'\s+', '', vehicle_no)

        # Usage characters for commercial vehicles
        usage_chars = '자아바사허하거너더러머버서어저고노도로모보소오조구누두루무부수우주'

        # Pattern 1: Region + digits + usage + digits (commercial)
        # Example: 강원70자1234
        pattern1 = rf'^([가-힣]+)(\d{{2,3}})([{usage_chars}])(\d{{4}})$'
        match = re.match(pattern1, normalized)
        if match:
            region, num1, usage, num2 = match.groups()
            return f"{region}{num1}{usage}{num2}"

        # Pattern 2: digits + Korean + digits (new format)
        # Example: 123가4567
        pattern2 = r'^(\d{2,3})([가-힣])(\d{4})$'
        match = re.match(pattern2, normalized)
        if match:
            num1, char, num2 = match.groups()
            return f"{num1}{char}{num2}"

        # Pattern 3: Region + digits (missing usage char) - try to infer
        # Example: 강원701234 -> 강원70자1234 (assume '자' for commercial buses)
        pattern3 = r'^([가-힣]{2,4})(\d{2})(\d{4})$'
        match = re.match(pattern3, normalized)
        if match:
            region, num1, num2 = match.groups()
            # Infer usage character based on context (70 series is typically commercial)
            if num1 in ('70', '71', '72', '73', '74', '75', '76', '77', '78', '79'):
                return f"{region}{num1}자{num2}"  # Assume '자' for commercial
            else:
                return f"{region}{num1}가{num2}"  # Assume '가' for others

        # Pattern 4: Region + usage + digits (missing 2-digit number)
        # Example: 제주아3140 -> 제주79아3140 (infer 79 for 아 usage in Jeju commercial)
        pattern4 = rf'^([가-힣]{{2,4}})([{usage_chars}])(\d{{4}})$'
        match = re.match(pattern4, normalized)
        if match:
            region, usage, num2 = match.groups()
            # Infer the 2-digit number based on region and usage character
            inferred_num = self._infer_vehicle_series(region, usage)
            if inferred_num:
                logging.info(f"Inferred vehicle series: {region}{inferred_num}{usage}{num2}")
                return f"{region}{inferred_num}{usage}{num2}"

        # If no pattern matches, just return cleaned version
        return normalized if len(normalized) >= 7 else None

    def _infer_vehicle_series(self, region, usage):
        """
        Infer the 2-digit vehicle series number based on region and usage character.

        Commercial vehicle series:
        - 70-79: Commercial buses (영업용 버스)
        - 아: Typically 79 series for commercial buses in many regions
        - 자: Typically 70-78 series for commercial vehicles
        """
        # Usage character to series mapping for commercial vehicles
        # 아 (rental/commercial) is typically 79 series
        # 자 (commercial) is typically 70 series
        usage_to_series = {
            '아': '79',  # 영업용 (rental/commercial)
            '자': '70',  # 영업용 (commercial)
            '바': '70',  # 영업용
            '사': '70',  # 영업용
        }

        return usage_to_series.get(usage)

    def _extract_vehicle_no_enhanced(self, text):
        """
        Enhanced vehicle number extraction with multiple patterns.
        """
        if not text:
            return None

        # Remove excessive whitespace but keep single spaces for pattern matching
        clean_text = re.sub(r'\s+', ' ', text)

        # Korean regions
        regions = '서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주'

        # Usage characters
        usage = '자아바사허하거너더러머버서어저고노도로모보소오조구누두루무부수우주'

        # Patterns to try (ordered by specificity)
        patterns = [
            # Pattern 1: After label "자동차등록번호"
            rf'(?:\uc790\ub3d9\ucc28\ub4f1\ub85d\ubc88\ud638)[^\n]*?((?:{regions})\s*\d{{2,3}}\s*[{usage}]\s*\d{{4}})',
            # Pattern 2: After ① marker
            rf'\u2460[^\n]*?((?:{regions})\s*\d{{2,3}}\s*[{usage}]\s*\d{{4}})',
            # Pattern 3: Direct match - Region + number + usage + number
            rf'\b((?:{regions})\s*\d{{2,3}}\s*[{usage}]\s*\d{{4}})\b',
            # Pattern 4: New format - 3 digits + Korean + 4 digits
            r'\b(\d{3}\s*[가-힣]\s*\d{4})\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, clean_text, re.IGNORECASE)
            if match:
                raw_value = match.group(1)
                normalized = self._normalize_vehicle_no(raw_value)
                if normalized:
                    return normalized

        return None

    def _extract_vehicle_no_from_filename(self, filename):
        """
        Extract vehicle number from filename.
        Filenames often contain the vehicle number like:
        - 경북70자6310.pdf
        - 제주79자7052_page_1.png
        - 강원70자1016_전기.pdf
        """
        if not filename:
            return None

        # Korean regions
        regions = '서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주'

        # Usage characters for commercial vehicles
        usage = '자아바사허하거너더러머버서어저고노도로모보소오조구누두루무부수우주'

        # Patterns to extract from filename
        patterns = [
            # Pattern 1: Region + 2-3 digits + usage char + 4 digits
            rf'((?:{regions})\d{{2,3}}[{usage}]\d{{4}})',
            # Pattern 2: With potential spaces/underscores
            rf'((?:{regions})[\s_]*\d{{2,3}}[\s_]*[{usage}][\s_]*\d{{4}})',
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                raw_value = match.group(1)
                # Remove underscores/spaces
                raw_value = re.sub(r'[\s_]+', '', raw_value)
                normalized = self._normalize_vehicle_no(raw_value)
                if normalized and self._is_valid_vehicle_no(normalized):
                    return normalized

        return None

    def _is_valid_vehicle_no(self, vehicle_no):
        """
        Validate Korean vehicle registration number format.
        """
        if not vehicle_no or len(vehicle_no) < 7:
            return False

        # Korean regions
        regions = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
                   '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']

        # Usage characters
        usage_chars = '자아바사허하거너더러머버서어저고노도로모보소오조구누두루무부수우주가나다라마'

        # Pattern: Region + 2-3 digits + usage char + 4 digits
        pattern = rf'^({"|".join(regions)})\d{{2,3}}[{usage_chars}]\d{{4}}$'

        return bool(re.match(pattern, vehicle_no))

    def _extract_fuel_type_from_filename(self, filename):
        """
        Extract fuel type from filename.
        Examples:
        - 강원70자1016_전기.pdf -> Electric
        - 경북70자6347_CNG_자동차등록증.pdf -> CNG
        """
        if not filename:
            return None

        for korean, english in self.FUEL_TYPES.items():
            if korean in filename:
                return english

        return None

    def _extract_fuel_type_from_text(self, text):
        """
        Extract fuel type from OCR text by looking for keywords.
        Searches for fuel-related keywords in the document text.
        """
        if not text:
            return None

        # Check each fuel type's keywords
        for fuel_type, keywords in self.FUEL_OCR_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return fuel_type

        return None

    def _determine_fuel_type(self, filename, ocr_text, model_name=None, engine_type=None):
        """
        Determine fuel type using multiple sources (priority order):
        1. Filename (most reliable)
        2. OCR text keywords
        3. Model name hints (e.g., 'ELEC CITY' -> Electric)
        4. Engine type hints
        """
        # Priority 1: Filename
        fuel_from_filename = self._extract_fuel_type_from_filename(filename)
        if fuel_from_filename:
            return fuel_from_filename

        # Priority 2: Model name hints
        if model_name:
            model_upper = model_name.upper()
            if 'ELEC' in model_upper or '전기' in model_name:
                return 'Electric'
            if 'SMART' in model_upper and ('110' in model_upper or '120' in model_upper):
                return 'Electric'  # SMART110/120 are typically electric buses

        # Priority 3: Engine type hints
        if engine_type:
            engine_upper = engine_type.upper()
            # Electric motor codes often start with specific patterns
            if engine_upper.startswith('TED') or engine_upper.startswith('EM'):
                return 'Electric'
            if 'CNG' in engine_upper:
                return 'CNG'

        # Priority 4: OCR text
        fuel_from_text = self._extract_fuel_type_from_text(ocr_text)
        if fuel_from_text:
            return fuel_from_text

        return 'Unknown'

    def verify_document_type(self, text):
        """
        Verify if text contains vehicle registration certificate keywords.
        Handles PaddleOCR output where spaces may be inserted between characters.
        """
        if not text:
            return False

        # Remove all spaces/whitespace for matching
        clean_text = re.sub(r'\s+', '', text)

        # Keywords to check in clean (no-space) text
        keywords = [
            '\uc790\ub3d9\ucc28\ub4f1\ub85d\uc99d',   # 자동차등록증
            '\uc790\ub3d9\ucc28\ub4f1\ub85d\uc99d\uc11c', # 자동차등록증서
            '\ucc28\ub300\ubc88\ud638',                 # 차대번호
            '\ub4f1\ub85d\ubc88\ud638',                 # 등록번호
            '\uc790\ub3d9\ucc28\ub4f1\ub85d',           # 자동차등록
            '\ucc28\ub7c9\ub4f1\ub85d',                 # 차량등록
        ]

        for keyword in keywords:
            if keyword in clean_text:
                return True

        # Flexible spacing patterns (OCR may insert spaces between chars)
        flexible_patterns = [
            r'\uc790\s*\ub3d9\s*\ucc28\s*\ub4f1\s*\ub85d',  # 자 동 차 등 록
            r'\ucc28\s*\ub300\s*\ubc88\s*\ud638',             # 차 대 번 호
            r'\ub4f1\s*\ub85d\s*\ubc88\s*\ud638',             # 등 록 번 호
            r'\ucd5c\s*\ucd08\s*\ub4f1\s*\ub85d\s*\uc77c',    # 최 초 등 록 일
            r'\uc81c\s*\uc6d0\s*\uad00\s*\ub9ac',             # 제 원 관 리
            r'\uc6d0\s*\ub3d9\s*\uae30\s*\ud615\s*\uc2dd',    # 원 동 기 형 식
            r'\uc2b9\s*\ucc28\s*\uc815\s*\uc6d0',             # 승 차 정 원
        ]

        for pattern in flexible_patterns:
            if re.search(pattern, text):
                return True

        # Check for VIN-like pattern (17-char alphanumeric starting with K)
        if re.search(r'\b[K][A-Z0-9]{16}\b', clean_text):
            return True

        return False

