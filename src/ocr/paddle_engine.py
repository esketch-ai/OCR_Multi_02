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
            return {'text': '', 'lines': [], 'avg_confidence': 0.0, 'debug': 'engine disabled'}

        try:
            if self._api_version == '3.x':
                return self._detect_text_v3(image_path)
            else:
                return self._detect_text_v2(image_path)
        except Exception as e:
            import traceback
            logging.error(f"PaddleOCR detection failed: {e}")
            return {'text': '', 'lines': [], 'avg_confidence': 0.0, 'debug': f'exception: {e}\n{traceback.format_exc()}'}

    def _detect_text_v3(self, image_path):
        """PaddleOCR 3.x API using predict(). Handles multiple result formats."""
        result = self.ocr.predict(image_path)

        if not result:
            logging.warning("PaddleOCR 3.x predict() returned empty result")
            return {'text': '', 'lines': [], 'avg_confidence': 0.0, 'debug': 'predict() returned empty'}

        # Build debug info for diagnostics
        debug_info = []
        try:
            debug_info.append(f"result type={type(result).__name__}, len={len(result) if hasattr(result, '__len__') else 'N/A'}")
            first = result[0]
            debug_info.append(f"result[0] type={type(first).__name__}")
            if hasattr(first, '__dict__'):
                debug_info.append(f"result[0] attrs={list(first.__dict__.keys())[:10]}")
            if hasattr(first, 'json'):
                json_val = first.json
                if isinstance(json_val, dict):
                    debug_info.append(f"json keys={list(json_val.keys())}")
                    res = json_val.get('res', None)
                    if res is not None:
                        debug_info.append(f"json.res type={type(res).__name__}")
                        if isinstance(res, dict):
                            debug_info.append(f"json.res keys={list(res.keys())}")
                        elif isinstance(res, list) and res:
                            debug_info.append(f"json.res[0] type={type(res[0]).__name__}, len={len(res)}")
                else:
                    debug_info.append(f"json type={type(json_val).__name__}")
            # Check string representation of first item
            first_str = str(first)[:300]
            debug_info.append(f"str(result[0])[:300]={first_str}")
        except Exception as e:
            debug_info.append(f"debug error: {e}")

        logging.info(f"PaddleOCR 3.x debug: {'; '.join(debug_info)}")

        texts = []
        scores = []

        # Strategy 1: result[0].json['res'] with rec_texts/rec_scores (PaddleOCR 3.0-3.2)
        try:
            ocr_result = result[0]
            if hasattr(ocr_result, 'json') and isinstance(ocr_result.json, dict):
                res = ocr_result.json.get('res', {})
                if isinstance(res, dict):
                    t = res.get('rec_texts', [])
                    s = res.get('rec_scores', [])
                    if t:
                        texts, scores = list(t), list(s)
                        logging.info(f"Parsed with Strategy 1 (json.res): {len(texts)} lines")
        except Exception as e:
            logging.debug(f"Strategy 1 failed: {e}")

        # Strategy 2: result[0].json['res'] is a list of dicts with 'rec_text'/'rec_score'
        if not texts:
            try:
                ocr_result = result[0]
                if hasattr(ocr_result, 'json') and isinstance(ocr_result.json, dict):
                    res = ocr_result.json.get('res', [])
                    if isinstance(res, list):
                        for item in res:
                            if isinstance(item, dict):
                                t = item.get('rec_text', item.get('text', ''))
                                s = item.get('rec_score', item.get('score', item.get('confidence', 0.0)))
                                if t:
                                    texts.append(t)
                                    scores.append(float(s))
                        if texts:
                            logging.info(f"Parsed with Strategy 2 (json.res list): {len(texts)} lines")
            except Exception as e:
                logging.debug(f"Strategy 2 failed: {e}")

        # Strategy 3: result is list of (bbox, (text, confidence)) tuples (PaddleOCR 3.3+)
        if not texts:
            try:
                for item in result:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        text_info = item[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            texts.append(str(text_info[0]))
                            scores.append(float(text_info[1]))
                if texts:
                    logging.info(f"Parsed with Strategy 3 (bbox tuples): {len(texts)} lines")
            except Exception as e:
                logging.debug(f"Strategy 3 failed: {e}")

        # Strategy 4: result is generator/iterable of result objects with 'rec' attribute
        if not texts:
            try:
                for item in result:
                    if hasattr(item, 'rec'):
                        for rec in item.rec:
                            t = getattr(rec, 'text', '') or (rec[0] if isinstance(rec, (list, tuple)) else '')
                            s = getattr(rec, 'score', 0.0) or (rec[1] if isinstance(rec, (list, tuple)) and len(rec) > 1 else 0.0)
                            if t:
                                texts.append(str(t))
                                scores.append(float(s))
                if texts:
                    logging.info(f"Parsed with Strategy 4 (rec attr): {len(texts)} lines")
            except Exception as e:
                logging.debug(f"Strategy 4 failed: {e}")

        # All strategies failed
        if not texts:
            debug_str = '; '.join(debug_info)
            logging.warning(f"All parsing strategies failed. Debug: {debug_str}")
            return {'text': '', 'lines': [], 'avg_confidence': 0.0, 'debug': debug_str}

        full_text = '\n'.join(texts)
        lines = list(zip(texts, scores))
        avg_conf = sum(scores) / len(scores) if scores else 0.0

        logging.info(f"OCR extracted {len(texts)} text lines, avg confidence: {avg_conf:.3f}")
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
