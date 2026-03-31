
# -*- coding: utf-8 -*-
import re
import logging
from src.validator.vin_validator import correct_vin_ocr, compute_check_digit, is_valid_structure


class CarRegistrationParser:
    """
    Parser for Korean Vehicle Registration Certificate (자동차등록증).
    Uses both label-based and label-independent extraction for robustness.
    Handles cases where PaddleOCR misreads Korean labels as Chinese characters.
    """

    # Korean VIN prefixes (World Manufacturer Identifier)
    VIN_PREFIXES = ('KM', 'KL', 'KN', 'KP', 'LA', 'LF', 'LG', 'LJ', 'KMJ', 'KMH', 'KNA')

    # Fuel type mappings (Korean -> standardized Korean output)
    # Hydrogen keywords MUST be checked before Electric (수소전기 contains 전기)
    FUEL_TYPES = {
        '수소': '수소전기',
        '수소전기': '수소전기',
        'FCEV': '수소전기',
        '연료전지': '수소전기',
        'CNG': 'CNG',
        '천연가스': 'CNG',
        '압축천연가스': 'CNG',
        '디젤': '경유',
        '경유': '경유',
        'LPG': 'LPG',
        '엘피지': 'LPG',
        '액화석유가스': 'LPG',
        '하이브리드': '하이브리드',
        '전기': '전기',
        '휘발유': '휘발유',
        '가솔린': '휘발유',
    }

    # Fuel keywords for OCR text detection
    # Order matters: Hydrogen before LPG (수소전기 contains 전기),
    # CNG before others, Electric last
    FUEL_OCR_KEYWORDS = [
        ('수소전기', ['수소', 'FCEV', '연료전지', '수소전기', 'HYDROGEN', 'H2']),
        ('CNG', ['CNG', '천연가스', '압축천연가스']),
        ('경유', ['디젤', '경유']),
        ('LPG', ['LPG', '엘피지', '액화석유가스']),
        ('하이브리드', ['하이브리드', 'HEV', 'PHEV']),
        ('휘발유', ['휘발유', '가솔린']),
        ('전기', ['전기', 'EV', 'ELEC']),
    ]

    def __init__(self):
        # Label-based regex patterns (work when Korean OCR is correct)
        self.label_fields = {
            'vehicle_no': r'(?:\uc790\ub3d9\ucc28\ub4f1\ub85d\ubc88\ud638)[\s:]*([^\n]+?)(?=\s*(?:\u2461|\u2462|\ucc28\uc885)|\n|$)',
            'vin': r'\ucc28\ub300\ubc88\ud638[^\n]*?[\s\n:]*([A-Z0-9]{17})',
            'vehicle_type': r'\ucc28\s*\uc885[\s:]*([^\n]+?)(?=\s*(?:\u2462|\u2463|\ucc28\uba85)|\n|$)',
            'model_name': r'(?:\ucc28\s*\uba85|Model\s*Name)[\s:]*([^\n/]+)',
            'vehicle_format': r'(?:\ud615\s*\uc2dd|\ud615\uc2dd\s*\ubc0f\s*\ubaa8\ub378\uc5f0\ub3c4)(?:.|\n){0,10}?\b([A-Z0-9-]{5,})(?=\s|/|\n)',
            'model_year': r'(?:\uc5f0\s*\uc2dd|\ubaa8\ub378\uc5f0\ub3c4)(?:.|\n){0,100}?\b((?:19|20)\d{2})\b',
            'engine_type': r'\uc6d0\ub3d9\uae30\ud615\uc2dd(?:.|\n){0,50}?([A-Z0-9-]{3,})',
            'owner_name': None,  # Table layout: label and value on different lines, handled by fallback
            'registration_date': r'\ucd5c\ucd08\ub4f1\ub85d\uc77c.*?(\d{4}\s*[\uac00-\ud7a3.\-/]\s*\d{1,2}\s*[\uac00-\ud7a3.\-/]\s*\d{1,2})',
            'length_mm': r'\uae38\s*\uc774.{0,30}?(\d[\d,]{3,6})\s*(?:mm|\u339c)?',
            'width_mm': r'\ub108\s*\ube44.{0,30}?(\d[\d,]{3,5})\s*(?:mm|\u339c)?',
            'height_mm': r'\ub192\s*\uc774.{0,30}?(\d[\d,]{3,5})\s*(?:mm|\u339c)?',
            'total_weight_kg': r'\ucd1d\s*\uc911?\s*\ub7c9.{0,30}?(\d[\d,]{3,4})\s*(?:kg|\u338f)?',
            'passenger_capacity': r'\uc2b9\ucc28\uc815\uc6d0.{0,30}?(\d{1,3})\s*(?:\uba85|\uc778)?',
        }

        # VIN fallback patterns (label-independent)
        self.vin_patterns = [
            # After Korean label (차대번호) with newlines
            r'\ucc28\ub300\ubc88\ud638[^\n]*?[\s\n:]*([A-Z0-9]{17})',
            # After ⑥ marker (may be on different line)
            r'\u2465[\s\S]{0,30}?([A-Z0-9]{17})',
            r'\u2466[\s\S]{0,30}?([A-Z0-9]{17})',
            # After "⑥" followed by garbled text then VIN on next line
            r'\u2465[^\n]*\n[^\n]*?([A-Z0-9]{17})',
            # Direct Korean vehicle VIN prefixes (most reliable)
            r'\b(KMJ[A-Z0-9]{14})\b',  # Hyundai buses
            r'\b(KMH[A-Z0-9]{14})\b',  # Hyundai cars
            r'\b(KNA[A-Z0-9]{14})\b',  # Kia
            r'\b(KM[A-Z0-9]{15})\b',
            r'\b(KL[A-Z0-9]{15})\b',
            r'\b(KN[A-Z0-9]{15})\b',
            r'\b(KP[A-Z0-9]{15})\b',
            r'\b(LA[A-Z0-9]{15})\b',
            # Any 17-char VIN-valid string
            r'\b([A-HJ-NPR-Z0-9]{17})\b',
        ]

        # Vehicle number patterns
        self.vehicle_no_patterns = [
            r'(?:\uc790\ub3d9\ucc28\ub4f1\ub85d\ubc88\ud638)[\s:]*([^\n]+?)(?=\s*(?:\u2461|\u2462|\ucc28)|\n|$)',
            r'\u2460[^\n]*?[\s:]*([가-힣]+\s*\d+\s*[자아바사]?\s*\d+)',
            r'\b([가-힣]{2,4}\s*\d{2,3}\s*[자아바사]\s*\d{4})\b',
        ]

    def parse(self, text):
        """Parse OCR text using label-based patterns, then fill gaps with label-independent fallbacks."""
        if not text:
            return {}

        result = {}

        # Step 1: Try label-based extraction
        for field, pattern in self.label_fields.items():
            if pattern is None:
                result[field] = None
                continue
            match = re.search(pattern, text, re.DOTALL)
            if match:
                value = match.group(1).strip()
                if field == 'vin':
                    value = self._correct_vin_ocr(value.upper())
                if field == 'registration_date':
                    value = re.sub(r'[\ub144\uc6d4\uc77c\s]+', '-', value).strip('-')
                # Reject bad captures starting with circled numbers
                bullets = ('\u2460', '\u2461', '\u2462', '\u2463', '\u2464', '\u2465', '\u2466')
                if value and value.startswith(bullets):
                    value = ''
                result[field] = value if value else None
            else:
                result[field] = None

        # Step 2: Label-independent fallbacks for each field
        self._fallback_owner_name(result, text)
        self._fallback_vin(result, text)
        self._fallback_vehicle_no(result, text)
        self._fallback_model_name(result, text)
        self._fallback_model_year(result, text)
        self._fallback_dimensions(result, text)
        self._fallback_passenger_capacity(result, text)
        self._fallback_registration_date(result, text)
        self._fallback_engine_type(result, text)
        self._fallback_vehicle_format(result, text)

        return result

    def _fallback_owner_name(self, result, text):
        """Extract owner name from the line BELOW the ⑨성명(명칭) header.

        Vehicle registration certs have table layout:
          ⑨ 성명(명칭)    생년월일     ← header line
             삼환교통      -           ← value line (target)
        """
        if result.get('owner_name'):
            return

        lines = text.split('\n')

        # Find the header line containing 성명/명칭/소유자 or ⑨
        label_patterns = [
            r'\uc131\s*\uba85',   # 성명
            r'\uba85\s*\uce6d',   # 명칭
            r'\uc18c\s*\uc720\s*\uc790',  # 소유자
            r'\u2468',            # ⑨
        ]

        for i, line in enumerate(lines):
            is_label_line = any(re.search(p, line) for p in label_patterns)
            if not is_label_line:
                continue

            # Strategy 1: Value is on the NEXT line (table layout)
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Skip if next line is another label/header
                skip_words = ('생년월일', '주소', '주민등록', '사업자', '등록번호',
                              '성명', '명칭', '소유자', '전화')
                if next_line and not any(w in next_line for w in skip_words):
                    # Take the first meaningful segment
                    name = re.split(r'\s{2,}', next_line)[0].strip()
                    name = self._clean_owner_name(name)
                    # Must contain at least one Korean character
                    if name and len(name) >= 2 and re.search(r'[가-힣]', name):
                        result['owner_name'] = name
                        logging.info(f"Owner name via next-line: {name}")
                        return

            # Strategy 2: Value is on the same line after all headers
            # e.g., "⑨ 성명(명칭) 생년월일 삼환교통 1990.01.01"
            # Remove known header words and take first Korean word group
            cleaned = line
            for hp in label_patterns:
                cleaned = re.sub(hp, '', cleaned)
            # Remove other header words
            for hw in ('생년월일', '주소', '주민등록번호', '사업자번호'):
                cleaned = cleaned.replace(hw, '')
            # Remove parentheses, circled numbers, digits-only
            cleaned = re.sub(r'[\u2460-\u2469()（）\d.:]+', ' ', cleaned)
            cleaned = cleaned.strip()
            if cleaned:
                name = re.split(r'\s{2,}', cleaned)[0].strip()
                name = self._clean_owner_name(name)
                if name and len(name) >= 2 and re.search(r'[가-힣]', name):
                    result['owner_name'] = name
                    logging.info(f"Owner name via same-line cleanup: {name}")
                    return

    def _fallback_vin(self, result, text):
        """Extract VIN using label-independent patterns."""
        if result.get('vin') and len(result['vin']) == 17:
            return

        # Pre-process: build space-stripped version for matching VINs split by spaces
        text_nospace = re.sub(r'\s+', '', text)

        # First pass: exact 17-char VINs from original text
        for pattern in self.vin_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                vin = self._correct_vin_ocr(match.upper().strip())
                if self._is_valid_vin(vin):
                    result['vin'] = vin
                    logging.info(f"VIN via fallback: {vin}")
                    return

        # Second pass: try space-stripped text (PaddleOCR splits VIN with spaces)
        for pattern in self.vin_patterns:
            matches = re.findall(pattern, text_nospace, re.IGNORECASE)
            for match in matches:
                vin = self._correct_vin_ocr(match.upper().strip())
                if self._is_valid_vin(vin):
                    result['vin'] = vin
                    logging.info(f"VIN via space-stripped fallback: {vin}")
                    return

        # Third pass: near-valid VINs (15-17 chars with Korean prefix)
        near_vin_patterns = [
            r'\b(K[A-Z0-9]{14,16})\b',
            r'\b(L[A-Z0-9]{14,16})\b',
        ]
        for src in (text, text_nospace):
            for pattern in near_vin_patterns:
                matches = re.findall(pattern, src, re.IGNORECASE)
                for match in matches:
                    vin = self._correct_vin_ocr(match.upper().strip())
                    if 15 <= len(vin) <= 17 and vin.isalnum():
                        if not any(c in vin for c in 'IOQ'):
                            result['vin'] = vin
                            logging.info(f"VIN via near-match ({len(vin)} chars): {vin}")
                            return

    def _fallback_vehicle_no(self, result, text):
        """Extract vehicle number using enhanced patterns."""
        if result.get('vehicle_no'):
            return

        clean_text = re.sub(r'\s+', ' ', text)
        regions = '서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주'
        usage = '자아바사허하거너더러머버서어저고노도로모보소오조구누두루무부수우주'

        patterns = [
            rf'(?:\uc790\ub3d9\ucc28\ub4f1\ub85d\ubc88\ud638)[^\n]*?((?:{regions})\s*\d{{2,3}}\s*[{usage}]\s*\d{{4}})',
            rf'\u2460[^\n]*?((?:{regions})\s*\d{{2,3}}\s*[{usage}]\s*\d{{4}})',
            rf'\b((?:{regions})\s*\d{{2,3}}\s*[{usage}]\s*\d{{4}})\b',
            r'\b(\d{3}\s*[가-힣]\s*\d{4})\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, clean_text)
            if match:
                normalized = self._normalize_vehicle_no(match.group(1))
                if normalized:
                    result['vehicle_no'] = normalized
                    logging.info(f"Vehicle No via fallback: {normalized}")
                    return

    def _fallback_model_name(self, result, text):
        """Extract model name using fallback patterns.
        Handles Korean model names like 뉴슈퍼에어로시티초저상버스."""
        if result.get('model_name'):
            # Clean: strip trailing whitespace/circled numbers from label match
            name = result['model_name'].strip()
            name = re.sub(r'[\u2460-\u2469].*', '', name).strip()
            result['model_name'] = name if name else None
            if result['model_name']:
                return

        # After ④ marker (차명 is field ④ on the cert)
        patterns = [
            r'\u2463[\s:]*([^\n]{2,})',          # After ④
            r'(?:\ucc28\s*\uba85)[\s:]*([^\n]+)',  # 차명 label (flexible spacing)
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1).strip()
                # Remove trailing circled numbers or format codes
                name = re.sub(r'\s*[\u2460-\u2469].*', '', name).strip()
                name = re.sub(r'\s*[A-Z0-9-]{5,}$', '', name).strip()
                if len(name) >= 2:
                    result['model_name'] = name
                    logging.info(f"Model name via fallback: {name}")
                    return

        # Look for known Korean bus/vehicle model name patterns
        known_models = [
            r'((?:뉴\s*)?슈퍼\s*에어로\s*시티[^\n]*?버스)',
            r'((?:뉴\s*)?에어로\s*시티[^\n]*?버스)',
            r'(유니버스[^\n]*)',
            r'(그린시티[^\n]*)',
            r'(일렉시티[^\n]*)',
            r'(카운티[^\n]*)',
            r'(마이티[^\n]*)',
        ]
        for pattern in known_models:
            match = re.search(pattern, text)
            if match:
                name = match.group(1).strip()
                result['model_name'] = name
                logging.info(f"Model name via known pattern: {name}")
                return

    def _fallback_model_year(self, result, text):
        """Extract model year using direct pattern matching."""
        if result.get('model_year'):
            return

        # Find all 4-digit years (19xx or 20xx)
        years = re.findall(r'\b((?:19|20)\d{2})\b', text)
        # Filter to reasonable vehicle years
        valid_years = [y for y in years if 1990 <= int(y) <= 2030]

        if valid_years:
            # Prefer year found after ⑤ marker
            match = re.search(r'\u2464[^\n]*?\b((?:19|20)\d{2})\b', text)
            if match:
                result['model_year'] = match.group(1)
            else:
                # Use the most common year, or the first valid one
                result['model_year'] = valid_years[0]
            logging.info(f"Model year via fallback: {result['model_year']}")

    def _fallback_dimensions(self, result, text):
        """Extract dimensions (length, width, height, weight).

        Strategy: Find the specs zone (after 제원관리번호 code like A08-1-xxx),
        then extract dimension values in document order.
        The cert always lists: length → width → height → weight → capacity.
        """
        fields = ('length_mm', 'width_mm', 'height_mm', 'total_weight_kg')
        if all(result.get(f) for f in fields):
            return

        lines = text.split('\n')

        # Find the specs zone: starts after specs number (e.g., A08-1-00059-xxx)
        spec_line_idx = 0
        for i, line in enumerate(lines):
            if re.search(r'[A-Z]\d{2}-\d-\d{4,}', line):
                spec_line_idx = i
                break

        # Excluded values: years, dates, page dimensions
        excluded = set()
        for y in range(1990, 2035):
            excluded.add(y)
        excluded.add(210)
        excluded.add(297)

        # Collect mm/kg-suffixed values from FULL TEXT (not per-line)
        # because PaddleOCR often puts the number and unit on separate lines
        spec_text = '\n'.join(lines[spec_line_idx:])

        mm_with_unit = []
        for m in re.finditer(r'(\d[\d,]{3,6})\s*(?:mm|\u339c)', spec_text, re.IGNORECASE):
            v = int(m.group(1).replace(',', ''))
            if v not in excluded and 1000 <= v <= 20000:
                mm_with_unit.append(v)

        kg_with_unit = []
        for m in re.finditer(r'(\d[\d,]{3,4})\s*(?:kg|\u338f)', spec_text, re.IGNORECASE):
            v = int(m.group(1).replace(',', ''))
            if 1000 <= v <= 99999:
                kg_with_unit.append(v)

        # Collect engine displacement and power values to exclude from dimensions
        # These appear near cc, Ps/rpm, rpm markers in the spec section
        engine_values = set()
        # Method 1: number followed by cc/c0
        for m in re.finditer(r'(\d[\d,]{3,6})\s*(?:cc|c0|c\b)', spec_text, re.IGNORECASE):
            engine_values.add(int(m.group(1).replace(',', '')))
        # Method 2: Find lines with cc/Ps/rpm markers
        # Exclude numbers AFTER marker (displacement/power values follow their labels)
        spec_lines = lines[spec_line_idx:]
        for i, line in enumerate(spec_lines):
            stripped = line.strip().lower()
            is_cc_marker = stripped in ('cc', 'c0')
            is_power_marker = 'ps/rpm' in stripped or 'ps/' in stripped

            if is_cc_marker or is_power_marker:
                # Exclude standalone numbers appearing AFTER this marker (within 5 lines)
                for j in range(i + 1, min(len(spec_lines), i + 5)):
                    adj = spec_lines[j].strip()
                    m = re.match(r'^(\d{4,6})$', adj)
                    if m:
                        engine_values.add(int(m.group(1)))
                        break  # Only the first number after marker

            # Also handle "number/number" format (like 290/2000 Ps/rpm)
            for m2 in re.finditer(r'(\d+)/(\d+)', line):
                if any(u in line.lower() for u in ('ps', 'rpm')):
                    engine_values.add(int(m2.group(1)))
                    engine_values.add(int(m2.group(2)))

        unique_mm = list(dict.fromkeys(mm_with_unit))
        unique_kg = list(dict.fromkeys(kg_with_unit))

        if len(unique_mm) >= 3:
            # Trust document order: length, width, height
            if not result.get('length_mm'):
                result['length_mm'] = str(unique_mm[0])
            if not result.get('width_mm'):
                result['width_mm'] = str(unique_mm[1])
            if not result.get('height_mm'):
                result['height_mm'] = str(unique_mm[2])
        else:
            # Use sequential order from spec zone
            spec_nums = self._extract_spec_zone_numbers(lines, spec_line_idx, excluded)

            # Remove kg-tagged values and engine values from dimension candidates
            exclude_from_dims = set(unique_kg) | engine_values
            dim_nums = [v for v in spec_nums if v not in exclude_from_dims]

            # Assign by range with document order preference
            self._assign_dimensions_by_range(result, dim_nums)

        # Weight: explicit kg values are authoritative
        if unique_kg:
            result['total_weight_kg'] = str(max(unique_kg))
        elif not result.get('total_weight_kg'):
            # Fallback: from spec zone, pick value > 5000 not used as dimension
            spec_nums = self._extract_spec_zone_numbers(lines, spec_line_idx, excluded)
            used = {int(result.get(f, 0)) for f in ('length_mm', 'width_mm', 'height_mm') if result.get(f)}
            for v in spec_nums:
                if v >= 5000 and v not in used:
                    result['total_weight_kg'] = str(v)
                    break

        # Post-validation: sanity check dimensions
        self._validate_dimensions(result)

    def _extract_spec_zone_numbers(self, lines, start_idx, excluded):
        """Extract numbers from the spec zone in document order.

        The vehicle registration cert spec section lists values in fixed order:
        길이(length) → 너비(width) → 높이(height) → 총중량(weight) → 배기량(cc) → 출력(Ps/rpm)

        Returns list of numeric values in order of appearance (first 4-6 values).
        """
        nums = []
        for line in lines[start_idx:]:
            stripped = line.strip()
            # Skip non-numeric lines, codes, and dates
            if re.match(r'^[A-Z]', stripped):
                continue
            if re.search(r'(Ps|rpm|cc|kmyL|km/)', stripped, re.IGNORECASE):
                # This line has power/displacement/fuel efficiency — stop collecting dims
                # But first check if there's a bare number before the unit
                m = re.match(r'^(\d[\d,]{3,6})', stripped)
                if m:
                    v = int(m.group(1).replace(',', ''))
                    if v not in excluded and 1000 <= v <= 30000:
                        nums.append(v)
                continue

            # Match standalone number or number with mm/kg suffix
            m = re.match(r'^(\d[\d,]{3,6})\s*(?:mm|kg|\u339c|\u338f)?\s*$', stripped)
            if m:
                v = int(m.group(1).replace(',', ''))
                if v not in excluded and 1000 <= v <= 30000 and v not in nums:
                    nums.append(v)
                continue

            # Match number at start of line followed by other content
            m = re.match(r'^(\d[\d,]{3,6})\s*(?:mm|kg|\u339c|\u338f)', stripped)
            if m:
                v = int(m.group(1).replace(',', ''))
                if v not in excluded and 1000 <= v <= 30000 and v not in nums:
                    nums.append(v)

            if len(nums) >= 6:
                break

        return nums

    def _extract_bare_dimension_numbers(self, lines, start_idx, excluded):
        """Extract standalone numbers from the spec zone that could be dimensions.
        Filters out numbers embedded in codes (like T2116-46) or dates."""
        bare = []
        for line in lines[start_idx:]:
            stripped = line.strip()
            # Skip lines that look like codes (contain letters mixed with numbers)
            if re.match(r'^[A-Z]\d+[-]', stripped):
                continue
            # Match purely standalone number line
            m = re.match(r'^(\d[\d,]{3,6})\s*$', stripped)
            if m:
                v = int(m.group(1).replace(',', ''))
                if v not in excluded and 1000 <= v <= 25000:
                    bare.append(v)
                continue
            # Match number with mm/kg suffix (already handled, but capture bare too)
            for m2 in re.finditer(r'(?<![A-Za-z\-])(\d{4,6})(?![A-Za-z0-9\-])', stripped):
                v = int(m2.group(1))
                if v not in excluded and 1000 <= v <= 25000 and v not in bare:
                    # Skip if part of a date-like pattern (YYYYMMDD)
                    if len(m2.group(1)) == 8:
                        continue
                    bare.append(v)
        return list(dict.fromkeys(bare))  # dedupe

    def _assign_dimensions_by_range(self, result, values):
        """Assign dimension values based on typical vehicle ranges.

        Ranges for Korean commercial vehicles:
          Length: 5000-14000mm (buses/trucks)
          Width:  1800-2600mm
          Height: 2500-3800mm (buses), 1300-2000mm (cars)
        """
        if not values:
            return

        used = set()

        if not result.get('length_mm'):
            for v in sorted(values, reverse=True):
                if v >= 5000:
                    result['length_mm'] = str(v)
                    used.add(v)
                    break

        remaining = [v for v in values if v not in used]

        if not result.get('width_mm'):
            for v in sorted(remaining):
                if 1800 <= v <= 2700:
                    result['width_mm'] = str(v)
                    used.add(v)
                    break

        remaining = [v for v in values if v not in used]

        if not result.get('height_mm'):
            for v in remaining:
                if 1300 <= v <= 4000:
                    result['height_mm'] = str(v)
                    used.add(v)
                    break

    def _validate_dimensions(self, result):
        """Post-validation: fix obviously wrong dimension assignments."""
        length = int(result['length_mm']) if result.get('length_mm') else None
        width = int(result['width_mm']) if result.get('width_mm') else None
        height = int(result['height_mm']) if result.get('height_mm') else None
        weight = int(result['total_weight_kg']) if result.get('total_weight_kg') else None

        # Width should be <= 2700mm
        if width and width > 2700 and height and height <= 2700:
            result['width_mm'], result['height_mm'] = result['height_mm'], result['width_mm']
            width, height = height, width

        # Height should be <= 4000mm; if larger, it's probably weight
        if height and height > 4000:
            if not weight:
                result['total_weight_kg'] = result['height_mm']
            result['height_mm'] = None

        # Weight should be max 5 digits (<=99999 kg)
        if weight and weight > 99999:
            result['total_weight_kg'] = None

        # Length should be > width and > height
        if length and width and length < width:
            result['length_mm'], result['width_mm'] = result['width_mm'], result['length_mm']

    def _fallback_passenger_capacity(self, result, text):
        """Extract passenger capacity from circled number markers or standalone patterns."""
        if result.get('passenger_capacity'):
            # Validate existing value
            val = int(result['passenger_capacity'])
            if val < 1 or val > 100:
                result['passenger_capacity'] = None
            else:
                return

        # Pattern 1: 승차정원 label + number + 명/인
        match = re.search(r'\uc2b9\ucc28\uc815\uc6d0.{0,30}?(\d{1,3})\s*(?:\uba85|\uc778)', text)
        if match:
            val = int(match.group(1))
            if 1 <= val <= 100:
                result['passenger_capacity'] = str(val)
                return

        # Pattern 2: number + 명 (but NOT preceded by digits — avoid matching "2020명")
        for match in re.finditer(r'(?<!\d)(\d{1,3})\s*(?:\uba85|\uc778)', text):
            val = int(match.group(1))
            # For buses, capacity is typically 20-80; for cars 2-9
            if 2 <= val <= 100:
                result['passenger_capacity'] = str(val)
                return

        # Pattern 3: standalone number in spec zone after weight values
        lines = text.split('\n')
        weight_found = False
        for line in lines:
            if re.search(r'\d+\s*(?:kg|\u338f)', line, re.IGNORECASE):
                weight_found = True
            if weight_found:
                match = re.match(r'^\s*(\d{1,3})\s*$', line)
                if match:
                    val = int(match.group(1))
                    # Bus capacity: 20-80, car: 2-9
                    if 2 <= val <= 100:
                        result['passenger_capacity'] = str(val)
                        return

    def _fallback_registration_date(self, result, text):
        """Extract registration date from date patterns in text.
        Prefers dates near '최초등록일' label over generic earliest date."""
        if result.get('registration_date'):
            return

        date_pattern = r'(\d{4})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})'

        # Strategy 1: Date near '최초등록일' or '등록일' label (within 50 chars)
        label_patterns = [
            r'최\s*초\s*등\s*록\s*일',
            r'등\s*록\s*일',
        ]
        for lp in label_patterns:
            match = re.search(lp + r'.{0,50}?' + date_pattern, text, re.DOTALL)
            if match:
                y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
                if 1990 <= y <= 2030 and 1 <= m <= 12 and 1 <= d <= 31:
                    result['registration_date'] = f"{y}-{m:02d}-{d:02d}"
                    logging.info(f"Registration date via label proximity: {result['registration_date']}")
                    return

        # Strategy 2: Fallback to all dates, prefer earliest
        dates = []
        for match in re.finditer(date_pattern, text):
            y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if 2000 <= y <= 2030 and 1 <= m <= 12 and 1 <= d <= 31:
                dates.append(f"{y}-{m:02d}-{d:02d}")
        # Also try compact format
        for match in re.finditer(r'(\d{4})(\d{2})(\d{2})', text):
            y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if 2000 <= y <= 2030 and 1 <= m <= 12 and 1 <= d <= 31:
                dates.append(f"{y}-{m:02d}-{d:02d}")

        if dates:
            dates.sort()
            result['registration_date'] = dates[0]
            logging.info(f"Registration date via fallback: {dates[0]}")

    def _fallback_engine_type(self, result, text):
        """Extract engine type code from text."""
        if result.get('engine_type'):
            return

        # Look after ⑦ marker for engine code
        match = re.search(r'\u2466[^\n]*?[\s\n]*([A-Z0-9]{3,10})', text)
        if match:
            result['engine_type'] = match.group(1)
            return

        # General: find alphanumeric codes 3-10 chars (like C6AF, D6CC, etc.)
        # Engine codes are typically short uppercase alphanumeric
        codes = re.findall(r'\b([A-Z]\d[A-Z0-9]{1,8})\b', text)
        for code in codes:
            # Skip VIN (17 chars) and format codes (too long)
            if 3 <= len(code) <= 8:
                result['engine_type'] = code
                return

    def _fallback_vehicle_format(self, result, text):
        """Extract vehicle format/model code."""
        if result.get('vehicle_format'):
            return

        # Look after ⑤ marker for format code
        match = re.search(r'\u2464[^\n]*?[\s\n]*([A-Z0-9][A-Z0-9-]{4,})', text)
        if match:
            result['vehicle_format'] = match.group(1)
            return

    def parse_single(self, ocr_text, filename=None):
        """
        Parse OCR text and return complete result.
        Uses filename fallback for vehicle_no and fuel type.
        """
        result = self.parse(ocr_text) if ocr_text else {}

        # Fallback: extract vehicle_no from filename if OCR failed
        ocr_vehicle_no = result.get('vehicle_no')
        if filename and (not ocr_vehicle_no or not self._is_valid_vehicle_no(ocr_vehicle_no)):
            filename_vehicle_no = self._extract_vehicle_no_from_filename(filename)
            if filename_vehicle_no:
                logging.info(f"Vehicle No from filename: {filename_vehicle_no} (OCR was: '{ocr_vehicle_no}')")
                result['vehicle_no'] = filename_vehicle_no

        # Determine fuel type (OCR text first, then model/engine hints, then filename)
        fuel_type = self._determine_fuel_type(
            filename=filename,
            ocr_text=ocr_text,
            model_name=result.get('model_name'),
            engine_type=result.get('engine_type'),
        )
        result['fuel_type'] = fuel_type

        return result

    def _clean_owner_name(self, name):
        """Remove company type suffixes/prefixes from owner name.
        Handles all OCR variants: (주), 주), (추], 주삼환 → 삼환 etc."""
        if not name:
            return name
        # Step 1: Remove any bracket/paren variations with company chars inside
        # Handles: (주), [주], (주,추), 주), (추], etc.
        name = re.sub(r'[\(\[\（][주쥬추유사재합특][^)\]）]*[\)\]）]', '', name)
        # Step 2: Remove standalone text company markers
        markers = [
            r'주식회사', r'유한회사', r'유한책임회사',
            r'사단법인', r'재단법인',
            r'합자회사', r'합명회사', r'합동회사',
        ]
        for pat in markers:
            name = re.sub(pat, '', name)
        # Step 3: Remove leading company-type char with optional broken brackets
        # e.g., "주삼환교통" → "삼환교통", "추]삼환교통" → "삼환교통"
        name = re.sub(r'^[\(\[\（]?[주쥬추유사재합특][\)\]\）,]*\s*', '', name)
        # Step 4: Clean up leftover brackets and punctuation
        name = re.sub(r'^[\(\[\（\)\]\）,\s]+', '', name)
        name = re.sub(r'[,\s]+', ' ', name).strip()
        return name

    def _correct_vin_ocr(self, vin):
        """Fix common OCR misreads in VIN using ISO 3779 check digit verification."""
        return correct_vin_ocr(vin)

    def _is_valid_vin(self, vin):
        """Validate VIN structure per ISO 3779."""
        valid, _ = is_valid_structure(vin)
        return valid

    def _normalize_vehicle_no(self, vehicle_no):
        if not vehicle_no:
            return None

        normalized = re.sub(r'\s+', '', vehicle_no)
        usage_chars = '자아바사허하거너더러머버서어저고노도로모보소오조구누두루무부수우주'

        # Pattern 1: Region + digits + usage + digits (commercial)
        pattern1 = rf'^([가-힣]+)(\d{{2,3}})([{usage_chars}])(\d{{4}})$'
        match = re.match(pattern1, normalized)
        if match:
            region, num1, usage, num2 = match.groups()
            return f"{region}{num1}{usage}{num2}"

        # Pattern 2: digits + Korean + digits (new format)
        pattern2 = r'^(\d{2,3})([가-힣])(\d{4})$'
        match = re.match(pattern2, normalized)
        if match:
            num1, char, num2 = match.groups()
            return f"{num1}{char}{num2}"

        # Pattern 3: Region + 6 digits (missing usage char)
        pattern3 = r'^([가-힣]{2,4})(\d{2})(\d{4})$'
        match = re.match(pattern3, normalized)
        if match:
            region, num1, num2 = match.groups()
            if num1 in ('70', '71', '72', '73', '74', '75', '76', '77', '78', '79'):
                return f"{region}{num1}사{num2}"
            else:
                return f"{region}{num1}가{num2}"

        return normalized if len(normalized) >= 7 else None

    def _extract_vehicle_no_from_filename(self, filename):
        if not filename:
            return None

        regions = '서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주'
        usage = '자아바사허하거너더러머버서어저고노도로모보소오조구누두루무부수우주'

        patterns = [
            rf'((?:{regions})\d{{2,3}}[{usage}]\d{{4}})',
            rf'((?:{regions})[\s_]*\d{{2,3}}[\s_]*[{usage}][\s_]*\d{{4}})',
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                raw_value = re.sub(r'[\s_]+', '', match.group(1))
                normalized = self._normalize_vehicle_no(raw_value)
                if normalized and self._is_valid_vehicle_no(normalized):
                    return normalized

        return None

    def _is_valid_vehicle_no(self, vehicle_no):
        if not vehicle_no or len(vehicle_no) < 7:
            return False

        regions = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
                   '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
        usage_chars = '자아바사허하거너더러머버서어저고노도로모보소오조구누두루무부수우주가나다라마'
        pattern = rf'^({"|".join(regions)})\d{{2,3}}[{usage_chars}]\d{{4}}$'
        return bool(re.match(pattern, vehicle_no))

    def _extract_fuel_type_from_filename(self, filename):
        if not filename:
            return None
        for korean, english in self.FUEL_TYPES.items():
            if korean in filename:
                return english
        return None

    def _extract_fuel_type_from_text(self, text):
        """Extract fuel type from OCR text. Checks specific fuels (CNG, LPG, Diesel) before Electric."""
        if not text:
            return None

        for fuel_type, keywords in self.FUEL_OCR_KEYWORDS:
            for keyword in keywords:
                if keyword in text:
                    return fuel_type

        return None

    def _determine_fuel_type(self, filename, ocr_text, model_name=None, engine_type=None):
        """
        Determine fuel type. Priority:
        1. Filename keywords (most reliable)
        2. OCR text keywords (check specific fuels before Electric)
        3. Engine type hints
        4. Model name hints
        """
        # Priority 1: Filename
        fuel_from_filename = self._extract_fuel_type_from_filename(filename)
        if fuel_from_filename:
            return fuel_from_filename

        # Priority 2: OCR text (specific fuels checked BEFORE Electric)
        fuel_from_text = self._extract_fuel_type_from_text(ocr_text)
        if fuel_from_text:
            return fuel_from_text

        # Priority 3: Engine type hints
        if engine_type:
            engine_upper = engine_type.upper()
            if 'FCEV' in engine_upper or 'FC' in engine_upper:
                return '수소전기'
            if 'CNG' in engine_upper:
                return 'CNG'
            if engine_upper.startswith('TED') or engine_upper.startswith('EM'):
                return '전기'

        # Priority 4: Model name hints
        if model_name:
            model_upper = model_name.upper()
            if '수소' in model_name or 'FCEV' in model_upper or 'HYDROGEN' in model_upper:
                return '수소전기'
            if 'ELEC' in model_upper or '전기' in model_name:
                return '전기'
            if 'SMART' in model_upper and ('110' in model_upper or '120' in model_upper):
                return '전기'

        return ''

    def verify_document_type(self, text):
        """
        Verify if text contains vehicle registration certificate keywords.
        Accepts both Korean text and garbled text with structural markers.
        """
        if not text:
            return False

        clean_text = re.sub(r'\s+', '', text)

        # Korean keywords (when OCR reads Korean correctly)
        keywords = [
            '자동차등록증', '자동차등록증서', '차대번호',
            '등록번호', '자동차등록', '차량등록',
        ]
        for keyword in keywords:
            if keyword in clean_text:
                return True

        # Flexible spacing patterns
        flexible_patterns = [
            r'\uc790\s*\ub3d9\s*\ucc28\s*\ub4f1\s*\ub85d',
            r'\ucc28\s*\ub300\s*\ubc88\s*\ud638',
            r'\ub4f1\s*\ub85d\s*\ubc88\s*\ud638',
            r'\ucd5c\s*\ucd08\s*\ub4f1\s*\ub85d\s*\uc77c',
            r'\uc81c\s*\uc6d0\s*\uad00\s*\ub9ac',
            r'\uc6d0\s*\ub3d9\s*\uae30\s*\ud615\s*\uc2dd',
            r'\uc2b9\s*\ucc28\s*\uc815\s*\uc6d0',
        ]
        for pattern in flexible_patterns:
            if re.search(pattern, text):
                return True

        # VIN pattern (K-prefixed 17 chars)
        if re.search(r'\b[K][A-Z0-9]{16}\b', clean_text):
            return True

        # Structural markers: circled numbers ①-⑩ (registration cert has these)
        circled_count = len(re.findall(r'[\u2460-\u2469]', text))
        if circled_count >= 3:
            return True

        # Dimension pattern: multiple mm/kg values suggest vehicle spec document
        mm_count = len(re.findall(r'\d+\s*mm', text, re.IGNORECASE))
        kg_count = len(re.findall(r'\d+\s*kg', text, re.IGNORECASE))
        if mm_count >= 2 and kg_count >= 1:
            return True

        return False
