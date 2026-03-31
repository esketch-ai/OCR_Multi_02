# -*- coding: utf-8 -*-
"""
PP-Structure layout analysis engine for document structure recognition.
Detects tables, text regions, and key-value pairs in vehicle registration certificates.
Supports PaddleOCR 2.x (PPStructure) and 3.x (PPStructureV3) APIs.
"""
import logging
import gc

logger = logging.getLogger(__name__)


class LayoutEngine:
    """
    PP-Structure based layout analysis engine (singleton).
    Extracts document structure: tables (as HTML), text regions with bboxes.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if LayoutEngine._initialized:
            return

        self.engine = None
        self.enabled = False
        self._api_version = None

        try:
            self._init_engine()
        except ImportError:
            logger.warning("PP-Structure not available. Layout analysis disabled.")
        except Exception as e:
            logger.error(f"Failed to init PP-Structure: {e}", exc_info=True)
        finally:
            LayoutEngine._initialized = True

    def _init_engine(self):
        """Initialize PP-Structure engine, auto-detecting API version."""
        import paddleocr
        version = getattr(paddleocr, '__version__', '2.0.0')
        major_version = int(version.split('.')[0])

        if major_version >= 3:
            self._init_v3()
        else:
            self._init_v2()

    def _init_v2(self):
        """PaddleOCR 2.x: PPStructure class.
        Note: PP-Structure layout models only support 'en' and 'ch'.
        We use 'ch' because CJK layout models handle Korean document structure well.
        OCR text recognition within PP-Structure will use Chinese,
        but we only use the layout/table structure — actual text comes from PaddleOCR korean engine.
        """
        from paddleocr import PPStructure
        logger.info("Initializing PP-Structure (2.x API, lang=ch for layout)...")
        self.engine = PPStructure(
            layout=True,
            table=True,
            ocr=False,
            show_log=False,
            lang='ch',
        )
        self._api_version = '2.x'
        self.enabled = True
        logger.info("PP-Structure 2.x initialized.")

    def _init_v3(self):
        """PaddleOCR 3.x: PPStructureV3 or table pipeline."""
        try:
            from paddleocr import PPStructureV3
            logger.info("Initializing PP-StructureV3 (3.x API)...")
            self.engine = PPStructureV3()
            self._api_version = '3.x'
            self.enabled = True
            logger.info("PP-StructureV3 initialized.")
        except ImportError:
            # Fallback: some 3.x versions may not have PPStructureV3
            logger.warning("PPStructureV3 not found in 3.x. Layout analysis disabled.")

    def analyze(self, image_path):
        """
        Analyze document layout and extract structured regions.

        Args:
            image_path: Path to preprocessed image file

        Returns:
            dict: {
                'tables': [{'bbox': (x1,y1,x2,y2), 'html': str, 'cells': list}],
                'text_regions': [{'bbox': (x1,y1,x2,y2), 'text': str, 'confidence': float}],
                'raw_regions': list  # all detected regions with type info
            }
        """
        if not self.enabled or not self.engine:
            return {'tables': [], 'text_regions': [], 'raw_regions': []}

        try:
            if self._api_version == '3.x':
                return self._analyze_v3(image_path)
            else:
                return self._analyze_v2(image_path)
        except Exception as e:
            logger.error(f"Layout analysis failed: {e}", exc_info=True)
            return {'tables': [], 'text_regions': [], 'raw_regions': []}

    def _analyze_v2(self, image_path):
        """PP-Structure 2.x analysis."""
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Cannot read image: {image_path}")
            return {'tables': [], 'text_regions': [], 'raw_regions': []}

        result = self.engine(img)
        if not result:
            return {'tables': [], 'text_regions': [], 'raw_regions': []}

        tables = []
        text_regions = []

        for region in result:
            region_type = region.get('type', '').lower()
            bbox = tuple(region.get('bbox', [0, 0, 0, 0]))
            res = region.get('res', None)

            if region_type == 'table' and res:
                html = res.get('html', '') if isinstance(res, dict) else ''
                cells = self._parse_table_html(html)
                tables.append({
                    'bbox': bbox,
                    'html': html,
                    'cells': cells,
                })
            elif region_type in ('text', 'title', 'header'):
                # res is list of (box, (text, confidence))
                if isinstance(res, list):
                    for item in res:
                        try:
                            text_info = item[1] if len(item) >= 2 else item
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text_regions.append({
                                    'bbox': bbox,
                                    'text': str(text_info[0]),
                                    'confidence': float(text_info[1]),
                                    'region_type': region_type,
                                })
                        except (IndexError, TypeError, ValueError):
                            continue

        logger.info(f"Layout analysis: {len(tables)} tables, {len(text_regions)} text regions")
        return {'tables': tables, 'text_regions': text_regions, 'raw_regions': result}

    def _analyze_v3(self, image_path):
        """PP-StructureV3 analysis."""
        result = self.engine.predict(image_path)
        if not result:
            return {'tables': [], 'text_regions': [], 'raw_regions': []}

        tables = []
        text_regions = []

        # V3 returns different structure - adapt based on actual output
        try:
            for item in result:
                if hasattr(item, 'json'):
                    data = item.json if isinstance(item.json, dict) else {}
                elif isinstance(item, dict):
                    data = item
                else:
                    continue

                res_list = data.get('res', [])
                if isinstance(res_list, list):
                    for region in res_list:
                        if not isinstance(region, dict):
                            continue
                        region_type = region.get('type', '').lower()
                        bbox = tuple(region.get('bbox', [0, 0, 0, 0]))

                        if region_type == 'table':
                            html = region.get('html', '')
                            cells = self._parse_table_html(html)
                            tables.append({'bbox': bbox, 'html': html, 'cells': cells})
                        elif region_type in ('text', 'title', 'header'):
                            text_regions.append({
                                'bbox': bbox,
                                'text': region.get('text', ''),
                                'confidence': float(region.get('score', 0.0)),
                                'region_type': region_type,
                            })
        except Exception as e:
            logger.warning(f"V3 result parsing error: {e}", exc_info=True)

        logger.info(f"Layout V3: {len(tables)} tables, {len(text_regions)} text regions")
        return {'tables': tables, 'text_regions': text_regions, 'raw_regions': list(result)}

    @staticmethod
    def _parse_table_html(html):
        """
        Parse table HTML into list of cell dicts.
        Returns: [{'row': int, 'col': int, 'text': str}]
        """
        if not html:
            return []

        cells = []
        try:
            # Simple regex-based HTML table parser (no lxml dependency)
            import re
            rows = re.findall(r'<tr>(.*?)</tr>', html, re.DOTALL)
            for row_idx, row_html in enumerate(rows):
                col_idx = 0
                for cell_match in re.finditer(r'<t[dh][^>]*>(.*?)</t[dh]>', row_html, re.DOTALL):
                    cell_text = re.sub(r'<[^>]+>', '', cell_match.group(1)).strip()
                    if cell_text:
                        cells.append({
                            'row': row_idx,
                            'col': col_idx,
                            'text': cell_text,
                        })
                    col_idx += 1
        except Exception as e:
            logger.debug(f"Table HTML parse error: {e}")

        return cells

    @classmethod
    def cleanup(cls):
        """Cleanup PP-Structure resources."""
        if cls._instance and cls._instance.engine:
            del cls._instance.engine
            cls._instance.engine = None
        cls._instance = None
        cls._initialized = False
        gc.collect()
