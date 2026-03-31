# -*- coding: utf-8 -*-
import os


class Config:
    # Base directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

    # OCR Settings
    OCR_LANGUAGE = 'korean'
    PDF_DPI = int(os.getenv('PDF_DPI', '300'))
    MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '2048'))

    # Layout Analysis (PP-Structure)
    ENABLE_LAYOUT_ANALYSIS = os.getenv('ENABLE_LAYOUT_ANALYSIS', 'true').lower() == 'true'

    # Processing Settings
    GC_INTERVAL = int(os.getenv('GC_INTERVAL', '10'))

    @classmethod
    def ensure_dirs(cls):
        """Create output directory if it doesn't exist."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
