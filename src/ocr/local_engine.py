# -*- coding: utf-8 -*-
"""
Local OCR Engine using PaddleOCR only (no Google Vision API).
Supports Korean and English text recognition with zero cloud costs.
"""
import logging
import gc
import warnings

warnings.filterwarnings('ignore')

from src.ocr.paddle_engine import LocalPaddleEngine


class LocalOCREngine:
    """
    Local-only OCR engine using PaddleOCR.
    Provides the same interface as HybridOCREngine for compatibility.
    """
    _instance = None
    _initialized = False

    def __new__(cls, lang='korean'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, lang='korean'):
        # Only initialize once (singleton pattern)
        if LocalOCREngine._initialized:
            return

        self.lang = lang
        logging.info(f"Initializing LocalOCREngine (lang={lang})...")

        try:
            self.paddle_engine = LocalPaddleEngine(lang=lang, enable_paddle=True)
            if not self.paddle_engine.enabled:
                raise RuntimeError("PaddleOCR failed to initialize")
            logging.info("LocalOCREngine initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize LocalOCREngine: {e}")
            raise

        LocalOCREngine._initialized = True

    def detect_text(self, image_path):
        """
        Detect text using PaddleOCR.

        Returns:
            dict: {
                'text': str (full text),
                'avg_conf': float (average confidence),
                'lines': list of (text, confidence) tuples
            }
        """
        result = self.paddle_engine.detect_text(image_path)
        return {
            'text': result['text'],
            'avg_conf': result['avg_confidence'],
            'lines': result['lines']
        }

    def detect_text_hybrid(self, image_path):
        """
        Compatibility method - returns result in hybrid engine format.
        Uses PaddleOCR for both 'google' and 'paddle' slots for compatibility
        with existing parser code.
        """
        logging.info(f"Running Local OCR on {image_path}...")

        paddle_result = self.paddle_engine.detect_text(image_path)
        paddle_raw = self.paddle_engine.detect_text_raw(image_path)

        # Return in hybrid format for compatibility with existing parser
        return {
            'google': {
                'text': paddle_result['text'],
                'avg_conf': paddle_result['avg_confidence'],
                'annotation': None  # No annotation in local mode
            },
            'paddle': {
                'result': paddle_raw,
                'text_lines': [line[0] for line in paddle_result['lines']]
            }
        }

    @classmethod
    def cleanup(cls):
        """Cleanup resources and reset singleton."""
        if cls._instance:
            LocalPaddleEngine.cleanup()
            cls._instance = None
        cls._initialized = False
        gc.collect()
