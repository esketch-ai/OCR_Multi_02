# -*- coding: utf-8 -*-
"""Tests for layout_engine.py and form_parser.py"""
import unittest
import sys

sys.stdout.reconfigure(encoding='utf-8')


class TestFormParser(unittest.TestCase):
    """Test FormParser with mock layout analysis results."""

    def setUp(self):
        from src.parser.form_parser import FormParser
        self.parser = FormParser()

    def _make_table(self, html, bbox=(0, 0, 800, 600)):
        """Helper to create a table result dict."""
        return {'bbox': bbox, 'html': html, 'cells': []}

    def test_extract_vehicle_no_from_table(self):
        html = '<table><tr><td>자동차등록번호</td><td>서울 12가 3456</td></tr></table>'
        layout = {'tables': [self._make_table(html)], 'text_regions': []}
        fields = self.parser.parse_layout(layout)
        self.assertIn('vehicle_no', fields)
        self.assertIn('서울 12가 3456', fields['vehicle_no']['value'])

    def test_extract_vin_from_table(self):
        html = '<table><tr><td>차대번호</td><td>KMHD341DBNU123456</td></tr></table>'
        layout = {'tables': [self._make_table(html)], 'text_regions': []}
        fields = self.parser.parse_layout(layout)
        self.assertIn('vin', fields)
        self.assertEqual(fields['vin']['value'], 'KMHD341DBNU123456')

    def test_extract_dimensions_from_table(self):
        html = (
            '<table>'
            '<tr><td>길이</td><td>4,880</td><td>너비</td><td>1,860</td></tr>'
            '<tr><td>높이</td><td>1,475</td><td>총중량</td><td>1,990</td></tr>'
            '</table>'
        )
        layout = {'tables': [self._make_table(html)], 'text_regions': []}
        fields = self.parser.parse_layout(layout)
        self.assertEqual(fields['length_mm']['value'], '4880')
        self.assertEqual(fields['width_mm']['value'], '1860')
        self.assertEqual(fields['height_mm']['value'], '1475')
        self.assertEqual(fields['total_weight_kg']['value'], '1990')

    def test_extract_owner_name_from_table(self):
        html = '<table><tr><td>성명(명칭)</td><td>홍길동</td></tr></table>'
        layout = {'tables': [self._make_table(html)], 'text_regions': []}
        fields = self.parser.parse_layout(layout)
        self.assertIn('owner_name', fields)
        self.assertEqual(fields['owner_name']['value'], '홍길동')

    def test_extract_from_text_regions(self):
        regions = [
            {'bbox': (100, 100, 200, 130), 'text': '차명', 'confidence': 0.95, 'region_type': 'text'},
            {'bbox': (220, 100, 400, 130), 'text': '쏘나타', 'confidence': 0.92, 'region_type': 'text'},
        ]
        layout = {'tables': [], 'text_regions': regions}
        fields = self.parser.parse_layout(layout, img_height=800, img_width=600)
        self.assertIn('model_name', fields)
        self.assertEqual(fields['model_name']['value'], '쏘나타')

    def test_circled_number_labels_matched(self):
        html = '<table><tr><td>②차종</td><td>승용차</td></tr></table>'
        layout = {'tables': [self._make_table(html)], 'text_regions': []}
        fields = self.parser.parse_layout(layout)
        self.assertIn('vehicle_type', fields)
        self.assertEqual(fields['vehicle_type']['value'], '승용차')

    def test_empty_layout_returns_empty(self):
        layout = {'tables': [], 'text_regions': []}
        fields = self.parser.parse_layout(layout)
        self.assertEqual(fields, {})

    def test_registration_date_cleaned(self):
        html = '<table><tr><td>최초등록일</td><td>2020년 01월 15일</td></tr></table>'
        layout = {'tables': [self._make_table(html)], 'text_regions': []}
        fields = self.parser.parse_layout(layout)
        self.assertIn('registration_date', fields)
        self.assertIn('2020', fields['registration_date']['value'])

    def test_model_year_extracted(self):
        html = '<table><tr><td>연식</td><td>2023</td></tr></table>'
        layout = {'tables': [self._make_table(html)], 'text_regions': []}
        fields = self.parser.parse_layout(layout)
        self.assertEqual(fields['model_year']['value'], '2023')

    def test_passenger_capacity_digits_only(self):
        html = '<table><tr><td>승차정원</td><td>5명</td></tr></table>'
        layout = {'tables': [self._make_table(html)], 'text_regions': []}
        fields = self.parser.parse_layout(layout)
        self.assertEqual(fields['passenger_capacity']['value'], '5')

    def test_label_not_used_as_value(self):
        """Label cell followed by another label should not produce a field."""
        html = '<table><tr><td>차종</td><td>차명</td></tr></table>'
        layout = {'tables': [self._make_table(html)], 'text_regions': []}
        fields = self.parser.parse_layout(layout)
        # 차명 is also a label, so 차종 should not get 차명 as its value
        if 'vehicle_type' in fields:
            self.assertNotEqual(fields['vehicle_type']['value'], '차명')


class TestLayoutEngine(unittest.TestCase):
    """Test LayoutEngine initialization and graceful degradation."""

    def test_parse_table_html(self):
        from src.ocr.layout_engine import LayoutEngine
        cells = LayoutEngine._parse_table_html(
            '<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>'
        )
        self.assertEqual(len(cells), 4)
        self.assertEqual(cells[0], {'row': 0, 'col': 0, 'text': 'A'})
        self.assertEqual(cells[3], {'row': 1, 'col': 1, 'text': 'D'})

    def test_parse_empty_html(self):
        from src.ocr.layout_engine import LayoutEngine
        cells = LayoutEngine._parse_table_html('')
        self.assertEqual(cells, [])

    def test_parse_nested_tags_stripped(self):
        from src.ocr.layout_engine import LayoutEngine
        cells = LayoutEngine._parse_table_html(
            '<table><tr><td><b>Bold</b> text</td></tr></table>'
        )
        self.assertEqual(cells[0]['text'], 'Bold text')


if __name__ == '__main__':
    unittest.main()
