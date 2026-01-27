import re

class VINValidator:
    """
    Validates Vehicle Identification Numbers (VIN).
    Standard: ISO 3779
    """
    
    @staticmethod
    def validate(vin):
        if not vin:
            return False, "Empty VIN"
            
        vin = vin.upper().replace(" ", "")
        
        # Check length
        if len(vin) != 17:
             return False, f"Invalid length: {len(vin)}"
             
        # Check forbidden characters (I, O, Q are not allowed in VINs)
        if re.search(r'[IOQ]', vin):
            return False, "Contains forbidden characters (I, O, Q)"
            
        # Optional: Checksum validation (complex and varies by region, skipping for now)
        return True, "Valid"

    @staticmethod
    def calculate_similarity(vin1, vin2):
        """
        Simple similarity check (e.g., Levenshtein distance could be used here).
        For now using simple matching percentage.
        """
        if not vin1 or not vin2:
            return 0.0
            
        matches = sum(c1 == c2 for c1, c2 in zip(vin1, vin2))
        return matches / max(len(vin1), len(vin2))
