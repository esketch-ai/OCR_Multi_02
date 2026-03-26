# -*- coding: utf-8 -*-
"""
PaddleOCR engine with singleton pattern and memory optimization.
Compatible with both PaddleOCR 2.x and 3.x APIs.
"""
import logging
import gc
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


class LocalPaddleEngine:
    """
    PaddleOCR engine with singleton pattern and memory optimization.
    Supports Korean and English text recognition.
    Auto-detects PaddleOCR version and uses appropriate API.
    """
    _instance = None
    _initialized = False

    def __new__(cls, lang='korean', enable_paddle=True):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, lang='korean', enable_paddle=True):
        # Only initialize once (singleton pattern)
        if LocalPaddleEngine._initialized:
            return

        self.ocr = None
        self.enabled = enable_paddle
        self.lang = lang
        self._api_version = None  # '2.x' or '3.x'

        if not enable_paddle:
            logging.info("PaddleOCR disabled by configuration")
            LocalPaddleEngine._initialized = True
            return

        try:
            from paddleocr import PaddleOCR
            import paddleocr
            version = getattr(paddleocr, '__version__', '2.0.0')
            major_version = int(version.split('.')[0])

            if major_version >= 3:
                self._api_version = '3.x'
                logging.info(f"Initializing PaddleOCR {version} (3.x API, lang={lang})...")
                self.ocr = PaddleOCR(
                    lang=lang,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_detection_model_name='PP-OCRv5_mobile_det',
                    text_det_limit_side_len=1280,
                    text_det_limit_type='max',
                )
            else:
                self._api_version = '2.x'
                logging.info(f"Initializing PaddleOCR {version} (2.x API, lang={lang})...")
                self.ocr = PaddleOCR(
                    lang=lang,
                    use_angle_cls=False,
                    use_gpu=False,
                    det_limit_side_len=1280,
                    det_limit_type='max',
                )

            logging.info(f"PaddleOCR initialized successfully (API: {self._api_version}).")

        except ImportError:
            logging.warning("PaddleOCR not installed. Disabling Paddle engine.")
            self.ocr = None
            self.enabled = False
        except Exception as e:
            logging.error(f"Failed to init PaddleOCR: {e}")
            self.ocr = None
            self.enabled = False

        LocalPaddleEngine._initialized = True

    def detect_text(self, image_path):
        """
        Detect text in an image using PaddleOCR.
        Auto-selects API based on installed version.

        Returns:
            dict: {
                'text': str (full combined text),
                'lines': list of (text, confidence) tuples,
                'avg_confidence': float
            }
        """
        if not self.enabled or not self.ocr:
            return {'text': '', 'lines': [], 'avg_confidence': 0.0}

        try:
            if self._api_version == '3.x':
                return self._detect_text_v3(image_path)
            else:
                return self._detect_text_v2(image_path)
        except Exception as e:
            logging.error(f"PaddleOCR detection failed: {e}")
            return {'text': '', 'lines': [], 'avg_confidence': 0.0}

    def _detect_text_v3(self, image_path):
        """PaddleOCR 3.x API using predict()."""
        result = self.ocr.predict(image_path)

        if not result or len(result) == 0:
            return {'text': '', 'lines': [], 'avg_confidence': 0.0}

        ocr_result = result[0]
        res = ocr_result.json['res']

        texts = res.get('rec_texts', [])
        scores = res.get('rec_scores', [])

        if not texts:
            return {'text': '', 'lines': [], 'avg_confidence': 0.0}

        full_text = '\n'.join(texts)
        lines = list(zip(texts, scores))
        avg_conf = sum(scores) / len(scores) if scores else 0.0

        return {'text': full_text, 'lines': lines, 'avg_confidence': avg_conf}

    def _detect_text_v2(self, image_path):
        """PaddleOCR 2.x API using ocr()."""
        result = self.ocr.ocr(image_path, cls=False)

        if not result or not result[0]:
            return {'text': '', 'lines': [], 'avg_confidence': 0.0}

        texts = []
        scores = []
        for line in result[0]:
            text, score = line[1]
            texts.append(text)
            scores.append(score)

        if not texts:
            return {'text': '', 'lines': [], 'avg_confidence': 0.0}

        full_text = '\n'.join(texts)
        lines = list(zip(texts, scores))
        avg_conf = sum(scores) / len(scores) if scores else 0.0

        return {'text': full_text, 'lines': lines, 'avg_confidence': avg_conf}

    @classmethod
    def cleanup(cls):
        """Cleanup PaddleOCR resources"""
        if cls._instance and cls._instance.ocr:
            del cls._instance.ocr
            cls._instance.ocr = None
        cls._instance = None
        cls._initialized = False
        gc.collect()
