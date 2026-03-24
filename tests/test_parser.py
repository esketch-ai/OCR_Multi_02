# -*- coding: utf-8 -*-
import unittest
import sys
from src.parser.car_registration import CarRegistrationParser

# Ensure utf-8 output
sys.stdout.reconfigure(encoding='utf-8')

class TestCarRegistrationParser(unittest.TestCase):
    def setUp(self):
        self.parser = CarRegistrationParser()
        
        # Sample OCR text using purely ASCII unicode escapes
        # All Korean characters are replaced with slash-u-XXXX
        
        # The sample text below matches the structure of a Korean Vehicle Registration Certificate
        # including Vehicle No, VIN, Engine Type, Owner, Address, Date, Specs.
        
        self.sample_text = (
            "\\uc790\\ub3d9\\ucc28\\ub4f1\\ub85d\\uc99d\\n"
            "\\uc790\\ub3d9\\ucc28\\ub4f1\\ub85d\\ubc88\\ud638 : 12\\uac00 3456    \\ucc28\\uc885 : \\uc2b9\\uc6a9\\ucc28\\n"
            "\\ucc28\\ub300\\ubc88\\ud638 : KL1T1234567890123    \\uc6d0\\ub3d9\\uae30\\ud6cc\\uc2dd : G4FD\\n"
            "\\ucc28\\uba85 : \\uc3d8\\ub098\\ud0c0               \\uc5f0\\uc2dd : 2020\\n"
            "\\uc18c\\uc720\\uc790 : \\ud64d\\uae38\\ub3d9\\n"
            "\\uc0ac\\uc6a9\\ubcf8\\uac70\\uc9c0 : \\uc11c\\uc6b8\\ud2b9\\ubcc4\\uc2dc \\uac15\\ub0a8\\uad6c \\uc5ed\\uc0bc\\ub3d9 123-45\\n"
            "\\ucd5c\\ucd08\\ub4f1\\ub85d\\uc77c : 2020-01-15\\n"
            "\\uc81c\\uc6d0\\uad00\\ub9ac\\ubc88\\ud638 : 123-456-789"
        )
        
        self.decoded_text = self.sample_text.encode('latin1').decode('unicode_escape')

    def test_parse_vehicle_no(self):
        result = self.parser.parse(self.decoded_text)
        self.assertEqual(result['vehicle_no'], '12\uac00 3456')

    def test_parse_vin(self):
        result = self.parser.parse(self.decoded_text)
        self.assertEqual(result['vin'], 'KL1T1234567890123')
        
    def test_parse_engine_type(self):
        result = self.parser.parse(self.decoded_text)
        self.assertEqual(result['engine_type'], 'G4FD')

    def test_parse_owner(self):
        result = self.parser.parse(self.decoded_text)
        self.assertEqual(result['owner_name'], '\ud64d\uae38\ub3d9')
        
    def test_parse_address(self):
        result = self.parser.parse(self.decoded_text)
        # Expected address string
        expected = '\uc11c\uc6b8\ud2b9\ubcc4\uc2dc \uac15\ub0a8\uad6c \uc5ed\uc0bc\ub3d9 123-45'
        self.assertEqual(result['owner_address'], expected)

    def test_parse_date(self):
        result = self.parser.parse(self.decoded_text)
        self.assertEqual(result['registration_date'], '2020-01-15')
        
    def test_parse_specs(self):
        result = self.parser.parse(self.decoded_text)
        self.assertEqual(result['vehicle_specs'], '123-456-789')

    def test_parse_single_returns_fuel_type(self):
        """parse_single should determine fuel type from filename."""
        result = self.parser.parse_single(self.decoded_text, filename="강원70자1016_전기.pdf")
        self.assertEqual(result['fuel_type'], 'Electric')

    def test_parse_single_returns_all_fields(self):
        """parse_single should return all 13 output fields."""
        required = [
            'vehicle_no', 'owner_name', 'vin', 'model_name', 'model_year',
            'registration_date', 'vehicle_type', 'length_mm', 'width_mm',
            'height_mm', 'total_weight_kg', 'passenger_capacity', 'fuel_type',
        ]
        result = self.parser.parse_single(self.decoded_text, filename="test.pdf")
        for field in required:
            self.assertIn(field, result, f"Missing field: {field}")

    def test_parse_single_vehicle_no_from_filename(self):
        """parse_single should extract vehicle_no from filename as fallback."""
        text = "자동차등록증\n차대번호 : KL1T1234567890123"
        result = self.parser.parse_single(text, filename="경북70자6310.pdf")
        self.assertEqual(result['vehicle_no'], '경북70자6310')

if __name__ == '__main__':
    unittest.main()
