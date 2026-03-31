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

    @staticmethod
    def _poly_to_rect(poly):
        """Convert 4-point polygon to (x1, y1, x2, y2) bounding rectangle."""
        try:
            if isinstance(poly, (list, tuple)) and len(poly) >= 4:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                return (min(xs), min(ys), max(xs), max(ys))
        except (TypeError, IndexError):
            pass
        return None

    def _detect_text_v3(self, image_path):
        """PaddleOCR 3.x API using predict(). Handles multiple result formats.
        Returns text + bounding boxes (ocr_results) for coordinate-based extraction."""
        result = self.ocr.predict(image_path)

        if not result:
            logging.warning("PaddleOCR 3.x predict() returned empty result")
            return {'text': '', 'lines': [], 'ocr_results': [], 'avg_confidence': 0.0, 'debug': 'predict() returned empty'}

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
                            if isinstance(res[0], dict):
                                debug_info.append(f"json.res[0] keys={list(res[0].keys())}")
                else:
                    debug_info.append(f"json type={type(json_val).__name__}")
            first_str = str(first)[:300]
            debug_info.append(f"str(result[0])[:300]={first_str}")
        except Exception as e:
            debug_info.append(f"debug error: {e}")

        logging.info(f"PaddleOCR 3.x debug: {'; '.join(debug_info)}")

        texts = []
        scores = []
        bboxes = []  # parallel list of (x1,y1,x2,y2) or None

        # Strategy 1: result[0].json['res'] with rec_texts/rec_scores/dt_polys (PaddleOCR 3.0-3.2)
        try:
            ocr_result = result[0]
            if hasattr(ocr_result, 'json') and isinstance(ocr_result.json, dict):
                res = ocr_result.json.get('res', {})
                if isinstance(res, dict):
                    t = res.get('rec_texts', [])
                    s = res.get('rec_scores', [])
                    polys = res.get('dt_polys', res.get('det_boxes', []))
                    if t:
                        texts, scores = list(t), list(s)
                        for i in range(len(texts)):
                            poly = polys[i] if i < len(polys) else None
                            bboxes.append(self._poly_to_rect(poly) if poly is not None else None)
                        logging.info(f"Parsed with Strategy 1 (json.res): {len(texts)} lines, {sum(1 for b in bboxes if b)} bboxes")
        except Exception as e:
            logging.debug(f"Strategy 1 failed: {e}")

        # Strategy 2: result[0].json['res'] is a list of dicts with 'rec_text'/'rec_score'/'dt_poly'
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
                                poly = item.get('dt_poly', item.get('det_box', item.get('bbox', None)))
                                if t:
                                    texts.append(t)
                                    scores.append(float(s))
                                    bboxes.append(self._poly_to_rect(poly) if poly is not None else None)
                        if texts:
                            logging.info(f"Parsed with Strategy 2 (json.res list): {len(texts)} lines, {sum(1 for b in bboxes if b)} bboxes")
            except Exception as e:
                logging.debug(f"Strategy 2 failed: {e}")

        # Strategy 3: result is list of (bbox, (text, confidence)) tuples (PaddleOCR 3.3+)
        if not texts:
            try:
                for item in result:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        bbox_points = item[0]
                        text_info = item[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            texts.append(str(text_info[0]))
                            scores.append(float(text_info[1]))
                            bboxes.append(self._poly_to_rect(bbox_points))
                if texts:
                    logging.info(f"Parsed with Strategy 3 (bbox tuples): {len(texts)} lines, {sum(1 for b in bboxes if b)} bboxes")
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
                                bboxes.append(None)
                if texts:
                    logging.info(f"Parsed with Strategy 4 (rec attr): {len(texts)} lines")
            except Exception as e:
                logging.debug(f"Strategy 4 failed: {e}")

        # All strategies failed
        if not texts:
            debug_str = '; '.join(debug_info)
            logging.warning(f"All parsing strategies failed. Debug: {debug_str}")
            return {'text': '', 'lines': [], 'ocr_results': [], 'avg_confidence': 0.0, 'debug': debug_str}

        full_text = '\n'.join(texts)
        lines = list(zip(texts, scores))
        avg_conf = sum(scores) / len(scores) if scores else 0.0

        # Build ocr_results with bbox for coordinate-based extraction
        ocr_results = []
        for i, (t, s) in enumerate(zip(texts, scores)):
            bbox = bboxes[i] if i < len(bboxes) else None
            if bbox:
                ocr_results.append({'text': t, 'confidence': s, 'bbox': bbox})

        logging.info(f"OCR extracted {len(texts)} text lines, {len(ocr_results)} with bbox, avg confidence: {avg_conf:.3f}")
        return {'text': full_text, 'lines': lines, 'ocr_results': ocr_results, 'avg_confidence': avg_conf}

    def _detect_text_v2(self, image_path):
        """PaddleOCR 2.x API using ocr(). Returns text + bounding boxes."""
        result = self.ocr.ocr(image_path, cls=False)

        if not result or not result[0]:
            return {'text': '', 'lines': [], 'ocr_results': [], 'avg_confidence': 0.0}

        texts = []
        scores = []
        ocr_results = []  # (text, confidence, bbox) for coordinate-based extraction
        for line in result[0]:
            bbox_points = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text, score = line[1]
            texts.append(text)
            scores.append(score)

            # Convert 4-point bbox to (x1, y1, x2, y2) rectangle
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            ocr_results.append({
                'text': text,
                'confidence': score,
                'bbox': bbox,  # (x1, y1, x2, y2)
            })

        if not texts:
            return {'text': '', 'lines': [], 'ocr_results': [], 'avg_confidence': 0.0}

        full_text = '\n'.join(texts)
        lines = list(zip(texts, scores))
        avg_conf = sum(scores) / len(scores) if scores else 0.0

        return {'text': full_text, 'lines': lines, 'ocr_results': ocr_results, 'avg_confidence': avg_conf}

    @classmethod
    def cleanup(cls):
        """Cleanup PaddleOCR resources"""
        if cls._instance and cls._instance.ocr:
            del cls._instance.ocr
            cls._instance.ocr = None
        cls._instance = None
        cls._initialized = False
        gc.collect()
