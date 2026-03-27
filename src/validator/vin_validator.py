# -*- coding: utf-8 -*-
"""
VIN (Vehicle Identification Number) Validator
Standard: ISO 3779 / SAE J853

Structure (17 characters):
  Position 1-3:  WMI (World Manufacturer Identifier)
  Position 4-8:  VDS (Vehicle Descriptor Section)
  Position 9:    Check digit (0-9 or X)
  Position 10:   Model year code
  Position 11:   Plant code
  Position 12-17: Sequential production number (VIS tail)
"""
import logging

logger = logging.getLogger(__name__)

# Transliteration values for check digit calculation (ISO 3779)
_TRANSLITERATION = {
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
    'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'P': 7, 'R': 9,
    'S': 2, 'T': 3, 'U': 4, 'V': 5, 'W': 6, 'X': 7, 'Y': 8, 'Z': 9,
}

# Position weights for check digit calculation
_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]

# Model year codes (position 10)
_YEAR_CODES = {
    'A': 2010, 'B': 2011, 'C': 2012, 'D': 2013, 'E': 2014,
    'F': 2015, 'G': 2016, 'H': 2017, 'J': 2018, 'K': 2019,
    'L': 2020, 'M': 2021, 'N': 2022, 'P': 2023, 'R': 2024,
    'S': 2025, 'T': 2026, 'V': 2027, 'W': 2028, 'X': 2029, 'Y': 2030,
    '1': 2001, '2': 2002, '3': 2003, '4': 2004, '5': 2005,
    '6': 2006, '7': 2007, '8': 2008, '9': 2009,
}

# Valid characters in VIN (I, O, Q excluded)
_VALID_CHARS = set('ABCDEFGHJKLMNPRSTUVWXYZ0123456789')

# Korean manufacturer WMIs
KOREAN_WMIS = {
    'KMH': 'Hyundai (passenger)',
    'KMJ': 'Hyundai (bus/truck)',
    'KME': 'Hyundai (EV/special)',
    'KNA': 'Kia',
    'KNC': 'Kia (commercial)',
    'KNM': 'Renault Samsung',
    'KND': 'Kia (SUV)',
    'KPT': 'SsangYong',
    'KPA': 'SsangYong',
    'KL1': 'GM Korea (Chevrolet)',
    'KLA': 'Daewoo/GM Korea',
    'KLB': 'Daewoo/GM Korea',
    'KMF': 'Hyundai (medium truck)',
    'KMK': 'Hyundai (special)',
}


def compute_check_digit(vin):
    """
    Compute the VIN check digit (position 9) per ISO 3779.

    Args:
        vin: 17-character VIN string (position 9 can be anything)

    Returns:
        Expected check digit character ('0'-'9' or 'X'), or None if invalid
    """
    if not vin or len(vin) != 17:
        return None

    total = 0
    for i, char in enumerate(vin.upper()):
        if i == 8:  # Skip position 9 (check digit itself)
            continue
        if char.isdigit():
            value = int(char)
        elif char in _TRANSLITERATION:
            value = _TRANSLITERATION[char]
        else:
            return None  # Invalid character
        total += value * _WEIGHTS[i]

    remainder = total % 11
    return 'X' if remainder == 10 else str(remainder)


def decode_model_year(vin):
    """Decode model year from VIN position 10."""
    if not vin or len(vin) < 10:
        return None
    year_char = vin[9].upper()
    return _YEAR_CODES.get(year_char)


def decode_wmi(vin):
    """Decode World Manufacturer Identifier (positions 1-3)."""
    if not vin or len(vin) < 3:
        return None
    wmi = vin[:3].upper()
    return KOREAN_WMIS.get(wmi)


def correct_vin_ocr(vin):
    """
    Fix common OCR misreads in VIN.

    Strategy:
    1. Replace universally invalid chars (I→1, O→0, Q→0)
    2. If check digit fails, try single-char corrections to find valid VIN
    """
    if not vin:
        return vin

    vin = vin.upper().strip()

    # Step 1: Fix universally invalid VIN characters
    vin = vin.replace('O', '0').replace('I', '1').replace('Q', '0')

    # Step 2: If check digit passes, we're done
    if len(vin) == 17:
        expected = compute_check_digit(vin)
        actual = vin[8]
        if expected and actual == expected:
            return vin

        # Step 3: Try single-character corrections at each position
        # Only try ambiguous OCR pairs (S↔5, B↔8, Z↔2, G↔6)
        ambiguous_pairs = [
            ('5', 'S'), ('S', '5'),
            ('8', 'B'), ('B', '8'),
            ('2', 'Z'), ('Z', '2'),
            ('6', 'G'), ('G', '6'),
            ('0', 'D'), ('D', '0'),
        ]
        for pos in range(17):
            if pos == 8:
                continue  # Don't modify check digit position
            original_char = vin[pos]
            for from_char, to_char in ambiguous_pairs:
                if original_char == from_char:
                    candidate = vin[:pos] + to_char + vin[pos + 1:]
                    chk = compute_check_digit(candidate)
                    if chk and candidate[8] == chk:
                        logger.info(f"VIN OCR correction: pos {pos + 1} '{from_char}'→'{to_char}' ({vin}→{candidate})")
                        return candidate

    return vin


def is_valid_structure(vin):
    """
    Validate VIN structure per ISO 3779.

    Returns:
        (is_valid, message) tuple
    """
    if not vin:
        return False, "Empty VIN"

    vin = vin.upper().strip()

    # Length check
    if len(vin) != 17:
        return False, f"Invalid length: {len(vin)} (expected 17)"

    # Character set check
    invalid_chars = set(vin) - _VALID_CHARS
    if invalid_chars:
        return False, f"Invalid characters: {invalid_chars}"

    # Position 12-17 should be numeric for Korean vehicles (most manufacturers)
    serial = vin[11:17]
    if vin[0] == 'K' and not serial.isdigit():
        # Some Korean VINs have alphanumeric serial, so this is a warning not rejection
        pass

    return True, "Valid structure"


def validate_check_digit(vin):
    """
    Validate VIN check digit (position 9).

    Note: Check digit is mandatory for North American vehicles (FMVSS/CMVSS)
    but optional for vehicles in other markets. Korean domestic vehicles
    may not always follow check digit rules, but exported ones do.

    Returns:
        (is_valid, message) tuple
    """
    if not vin or len(vin) != 17:
        return False, "Cannot validate check digit"

    expected = compute_check_digit(vin)
    actual = vin[8]

    if expected is None:
        return False, "Cannot compute check digit (invalid characters)"

    if actual == expected:
        return True, f"Check digit valid ({actual})"
    else:
        return False, f"Check digit mismatch: expected '{expected}', got '{actual}'"


class VINValidator:
    """
    Validates Vehicle Identification Numbers (VIN).
    Standard: ISO 3779 / SAE J853

    Validation levels:
    1. Structure: length, character set
    2. Check digit: position 9 verification
    3. WMI: manufacturer identification
    """

    @staticmethod
    def validate(vin):
        """
        Full VIN validation.

        Returns:
            (is_valid, message) tuple
        """
        if not vin:
            return False, "Empty VIN"

        vin = vin.upper().replace(" ", "")

        # Structure validation
        valid, msg = is_valid_structure(vin)
        if not valid:
            return False, msg

        # Check digit validation
        chk_valid, chk_msg = validate_check_digit(vin)

        # WMI info
        manufacturer = decode_wmi(vin)
        year = decode_model_year(vin)

        info_parts = []
        if manufacturer:
            info_parts.append(manufacturer)
        if year:
            info_parts.append(f"MY{year}")

        if chk_valid:
            info = f"Valid ({', '.join(info_parts)})" if info_parts else "Valid"
            return True, info
        else:
            # Korean domestic vehicles may not use check digit
            # Accept with warning if structure is valid
            info = f"Valid (structure OK, {chk_msg}"
            if info_parts:
                info += f", {', '.join(info_parts)}"
            info += ")"
            return True, info

    @staticmethod
    def calculate_similarity(vin1, vin2):
        """Simple character-level similarity between two VINs."""
        if not vin1 or not vin2:
            return 0.0

        matches = sum(c1 == c2 for c1, c2 in zip(vin1, vin2))
        return matches / max(len(vin1), len(vin2))
