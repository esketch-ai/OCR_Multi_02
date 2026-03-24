# -*- coding: utf-8 -*-
import unittest
from unittest.mock import patch, MagicMock
from src.processor import process_single_file, process_batch, results_to_rows


class TestProcessSingleFile(unittest.TestCase):

    @patch('src.processor.get_ocr_engine')
    @patch('src.processor.ImagePreprocessor')
    def test_returns_dict_with_all_fields(self, mock_preprocessor_cls, mock_get_engine):
        mock_prep = MagicMock()
        mock_prep.load_image.return_value = ['/tmp/fake_image.jpg']
        mock_preprocessor_cls.return_value = mock_prep

        mock_engine = MagicMock()
        mock_engine.detect_text.return_value = {
            'text': '자동차등록증\n차대번호 : KL1T1234567890123',
            'lines': [('자동차등록증', 0.95), ('차대번호 : KL1T1234567890123', 0.92)],
            'avg_confidence': 0.93,
        }
        mock_get_engine.return_value = mock_engine

        result = process_single_file('/tmp/fake.jpg', 'fake.jpg')
        self.assertEqual(result['status'], 'success')
        self.assertIn('data', result)
        self.assertIn('vin', result['data'])

    @patch('src.processor.get_ocr_engine')
    @patch('src.processor.ImagePreprocessor')
    def test_returns_error_on_non_registration(self, mock_preprocessor_cls, mock_get_engine):
        mock_prep = MagicMock()
        mock_prep.load_image.return_value = ['/tmp/fake_image.jpg']
        mock_preprocessor_cls.return_value = mock_prep

        mock_engine = MagicMock()
        mock_engine.detect_text.return_value = {
            'text': 'This is a random document with no registration keywords',
            'lines': [('This is a random document', 0.9)],
            'avg_confidence': 0.9,
        }
        mock_get_engine.return_value = mock_engine

        result = process_single_file('/tmp/fake.jpg', 'fake.jpg')
        self.assertEqual(result['status'], 'skipped')


class TestProcessBatch(unittest.TestCase):

    @patch('src.processor.process_single_file')
    def test_batch_returns_list(self, mock_process):
        mock_process.return_value = {'status': 'success', 'data': {}, 'filename': 'a.jpg'}
        files = [('/tmp/a.jpg', 'a.jpg'), ('/tmp/b.jpg', 'b.jpg')]
        results = process_batch(files)
        self.assertEqual(len(results), 2)


class TestResultsToRows(unittest.TestCase):

    def test_converts_success_results(self):
        results = [
            {'status': 'success', 'filename': 'a.jpg', 'data': {
                'vehicle_no': '경북70자6310', 'owner_name': '홍길동', 'vin': 'KM8J12345678901234',
                'model_name': 'UNIVERSE', 'model_year': '2020', 'registration_date': '2020-01-15',
                'vehicle_type': '승용차', 'length_mm': '12000', 'width_mm': '2500',
                'height_mm': '3400', 'total_weight_kg': '18000', 'passenger_capacity': '45',
                'fuel_type': 'Diesel',
            }},
            {'status': 'skipped', 'filename': 'b.jpg', 'data': {}, 'message': 'Not a registration'},
        ]
        rows = results_to_rows(results)
        self.assertEqual(len(rows), 1)  # Only success rows
        self.assertEqual(rows[0][0], '경북70자6310')
        self.assertEqual(len(rows[0]), 13)  # 13 columns

    def test_empty_results(self):
        rows = results_to_rows([])
        self.assertEqual(rows, [])


if __name__ == '__main__':
    unittest.main()
