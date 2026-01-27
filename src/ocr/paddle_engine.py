import logging
import gc


class LocalPaddleEngine:
    """
    PaddleOCR engine with singleton pattern and memory optimization.
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

        if not enable_paddle:
            logging.info("PaddleOCR disabled by configuration")
            LocalPaddleEngine._initialized = True
            return

        try:
            logging.info("Initializing PaddleOCR (Local)...")
            from paddleocr import PaddleOCR

            # Optimized settings for memory efficiency
            self.ocr = PaddleOCR(
                use_angle_cls=False,  # Disable angle classification
                lang=lang,
                use_gpu=False,        # Force CPU to avoid GPU memory issues
                enable_mkldnn=False,  # Disable MKL-DNN to save memory
                show_log=False,       # Disable verbose logging
                det_db_unclip_ratio=1.5,  # Reduce detection sensitivity
            )
            logging.info("PaddleOCR Initialized successfully.")
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
        Returns:
            list of [ [ [x1,y1], [x2,y2], ... ], (text, confidence) ]
        """
        if not self.enabled or not self.ocr:
            return []

        try:
            # Try newer predict() method first (PaddleOCR 2.7+)
            if hasattr(self.ocr, 'predict'):
                result = self.ocr.predict(image_path)
            else:
                # Fallback to deprecated ocr() method
                result = self.ocr.ocr(image_path, cls=False)

            if not result or result[0] is None:
                return []

            return result[0]
        except TypeError as e:
            # Handle cls argument error - try without it
            if 'cls' in str(e):
                try:
                    if hasattr(self.ocr, 'predict'):
                        result = self.ocr.predict(image_path)
                    else:
                        result = self.ocr.ocr(image_path)
                    if not result or result[0] is None:
                        return []
                    return result[0]
                except Exception as e2:
                    logging.error(f"PaddleOCR detection failed (retry): {e2}")
                    return []
            logging.error(f"PaddleOCR detection failed: {e}")
            return []
        except Exception as e:
            logging.error(f"PaddleOCR detection failed: {e}")
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
