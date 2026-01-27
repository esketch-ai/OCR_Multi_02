import logging
import gc
from src.ocr.engine import OCREngine
from src.ocr.paddle_engine import LocalPaddleEngine


class HybridOCREngine:
    """
    Hybrid OCR Engine combining Google Vision and PaddleOCR.
    Uses singleton pattern for memory efficiency.
    """
    _instance = None
    _initialized = False

    def __new__(cls, enable_paddle=True):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, enable_paddle=True):
        # Only initialize once (singleton pattern)
        if HybridOCREngine._initialized:
            return

        self.enable_paddle = enable_paddle
        self.google_engine = OCREngine()

        if enable_paddle:
            try:
                self.paddle_engine = LocalPaddleEngine(enable_paddle=True)
            except Exception as e:
                logging.warning(f"Failed to initialize PaddleOCR: {e}. Continuing with Google only.")
                self.paddle_engine = None
                self.enable_paddle = False
        else:
            self.paddle_engine = None
            logging.info("PaddleOCR disabled by configuration")

        HybridOCREngine._initialized = True

    def detect_text_hybrid(self, image_path):
        """
        Runs both engines and returns a combined result object.
        """
        logging.info(f"Running Hybrid OCR on {image_path}...")

        # 1. Google Vision (primary)
        google_result = self.google_engine.detect_text(image_path)

        # Handle both 2-tuple and 3-tuple returns
        if len(google_result) == 3:
            google_text, google_avg_conf, google_annotation = google_result
        else:
            google_text, google_avg_conf = google_result
            google_annotation = None

        # 2. PaddleOCR (secondary, if enabled)
        paddle_result = []
        if self.enable_paddle and self.paddle_engine:
            try:
                paddle_result = self.paddle_engine.detect_text(image_path)
            except Exception as e:
                logging.warning(f"PaddleOCR failed for {image_path}: {e}")
                paddle_result = []

        # Combine into a structured dict
        hybrid_result = {
            'google': {
                'text': google_text,
                'avg_conf': google_avg_conf,
                'annotation': google_annotation
            },
            'paddle': {
                'result': paddle_result,  # list of [box, (text, score)]
                'text_lines': [line[1][0] for line in paddle_result] if paddle_result else []
            }
        }

        return hybrid_result

    @classmethod
    def cleanup(cls):
        """Cleanup resources and reset singleton."""
        if cls._instance:
            if cls._instance.paddle_engine:
                LocalPaddleEngine.cleanup()
            cls._instance = None
        cls._initialized = False
        gc.collect()
