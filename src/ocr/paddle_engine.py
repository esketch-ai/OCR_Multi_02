# -*- coding: utf-8 -*-
"""
PaddleOCR engine with singleton pattern and memory optimization.
Updated for PaddleOCR 3.3+ API.
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

        if not enable_paddle:
            logging.info("PaddleOCR disabled by configuration")
            LocalPaddleEngine._initialized = True
            return

        try:
            logging.info(f"Initializing PaddleOCR 3.3+ (lang={lang})...")
            from paddleocr import PaddleOCR

            # PaddleOCR 3.3+ configuration optimized for speed
            # Mobile detection + Korean recognition for best speed/accuracy balance
            self.ocr = PaddleOCR(
                lang=lang,
                use_doc_orientation_classify=False,  # Skip orientation detection
                use_doc_unwarping=False,             # Skip document unwarping
                use_textline_orientation=False,      # Skip textline orientation
                text_detection_model_name='PP-OCRv5_mobile_det',  # Mobile detection (4x faster)
                text_det_limit_side_len=1280,        # Limit image size for speed
                text_det_limit_type='max',
            )
            logging.info("PaddleOCR initialized successfully (optimized for speed).")

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
        Detect text in an image using PaddleOCR 3.3+.

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
            # PaddleOCR 3.3+ uses predict method
            result = self.ocr.predict(image_path)

            if not result or len(result) == 0:
                return {'text': '', 'lines': [], 'avg_confidence': 0.0}

            # Extract data from OCRResult object
            ocr_result = result[0]
            res = ocr_result.json['res']

            texts = res.get('rec_texts', [])
            scores = res.get('rec_scores', [])

            if not texts:
                return {'text': '', 'lines': [], 'avg_confidence': 0.0}

            # Combine texts
            full_text = '\n'.join(texts)

            # Create lines with confidence
            lines = list(zip(texts, scores))

            # Calculate average confidence
            avg_conf = sum(scores) / len(scores) if scores else 0.0

            return {
                'text': full_text,
                'lines': lines,
                'avg_confidence': avg_conf
            }

        except Exception as e:
            logging.error(f"PaddleOCR detection failed: {e}")
            return {'text': '', 'lines': [], 'avg_confidence': 0.0}

    def detect_text_raw(self, image_path):
        """
        Returns raw PaddleOCR result for advanced processing.
        Compatible with older code expecting list of [box, (text, score)].
        """
        if not self.enabled or not self.ocr:
            return []

        try:
            result = self.ocr.predict(image_path)

            if not result or len(result) == 0:
                return []

            ocr_result = result[0]
            res = ocr_result.json['res']

            texts = res.get('rec_texts', [])
            scores = res.get('rec_scores', [])
            boxes = res.get('rec_polys', [])

            # Convert to old format: [[box], (text, score)]
            raw_result = []
            for i, (text, score) in enumerate(zip(texts, scores)):
                box = boxes[i] if i < len(boxes) else [[0, 0], [0, 0], [0, 0], [0, 0]]
                raw_result.append([box, (text, score)])

            return raw_result

        except Exception as e:
            logging.error(f"PaddleOCR raw detection failed: {e}")
            return []

    @classmethod
    def cleanup(cls):
        """Cleanup PaddleOCR resources"""
        if cls._instance and cls._instance.ocr:
            del cls._instance.ocr
            cls._instance.ocr = None
        cls._instance = None
        cls._initialized = False
        gc.collect()
