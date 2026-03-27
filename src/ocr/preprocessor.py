# -*- coding: utf-8 -*-
"""
Image preprocessor with OCR quality enhancement.
Ported from OCR_Vehicle_02: CLAHE, deskew, denoise.
"""
import os
import gc
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Image enhancement disabled.")

try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


class ImagePreprocessor:
    """
    Image preprocessor with OCR quality enhancement.

    Enhancement pipeline (from OCR_Vehicle_02):
    1. PDF → Image (300 DPI)
    2. Deskew (Hough Line Transform)
    3. Denoise (Gaussian Blur)
    4. Contrast enhancement (CLAHE on L channel)
    """
    DEFAULT_DPI = 300           # 300 DPI for better OCR (was 200)
    MAX_IMAGE_SIZE = 2048
    JPEG_QUALITY = 92           # Higher quality for OCR
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_GRID_SIZE = 8
    GAUSSIAN_KERNEL = 3

    def __init__(self, dpi=None, max_size=None):
        self.dpi = dpi or self.DEFAULT_DPI
        self.max_size = max_size or self.MAX_IMAGE_SIZE

    def _resize_image_if_needed(self, img):
        """Resize image if it exceeds maximum dimension."""
        width, height = img.size
        if width <= self.max_size and height <= self.max_size:
            return img

        if width > height:
            new_width = self.max_size
            new_height = int(height * (self.max_size / width))
        else:
            new_height = self.max_size
            new_width = int(width * (self.max_size / height))

        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _enhance_image(self, img_path):
        """Apply OCR enhancement pipeline: deskew → denoise → CLAHE.

        Args:
            img_path: Path to image file

        Returns:
            Path to enhanced image (may be same path if no enhancement needed)
        """
        if not CV2_AVAILABLE:
            return img_path

        try:
            img = cv2.imread(img_path)
            if img is None:
                return img_path

            # Step 1: Deskew (correct rotation from scanning)
            img = self._deskew(img)

            # Step 2: Denoise (Gaussian blur)
            img = self._denoise(img)

            # Step 3: CLAHE contrast enhancement
            img = self._enhance_contrast(img)

            # Save enhanced image
            base, ext = os.path.splitext(img_path)
            enhanced_path = f"{base}_enhanced.jpg"
            cv2.imwrite(enhanced_path, img, [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])

            return enhanced_path

        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return img_path

    def _deskew(self, img):
        """Correct image skew using Hough Line Transform."""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=100, minLineLength=100, maxLineGap=10
            )

            if lines is None:
                return img

            # Collect angles of near-horizontal lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 15:  # Only near-horizontal lines
                    angles.append(angle)

            if not angles:
                return img

            median_angle = np.median(angles)

            # Only rotate if skew is significant (> 0.5 degrees)
            if abs(median_angle) < 0.5:
                return img

            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                img, M, (w, h),
                borderMode=cv2.BORDER_REPLICATE
            )

            logger.info(f"Deskew applied: {median_angle:.2f}°")
            return rotated

        except Exception as e:
            logger.warning(f"Deskew failed: {e}")
            return img

    def _denoise(self, img):
        """Remove noise with Gaussian blur."""
        try:
            return cv2.GaussianBlur(img, (self.GAUSSIAN_KERNEL, self.GAUSSIAN_KERNEL), 0)
        except Exception:
            return img

    def _enhance_contrast(self, img):
        """Enhance contrast using CLAHE on L channel (preserves color)."""
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=self.CLAHE_CLIP_LIMIT,
                tileGridSize=(self.CLAHE_GRID_SIZE, self.CLAHE_GRID_SIZE)
            )
            l_enhanced = clahe.apply(l_channel)

            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

            return enhanced

        except Exception as e:
            logger.warning(f"CLAHE failed: {e}")
            return img

    def load_image(self, image_path, max_pages=None):
        """Load and enhance image/PDF for OCR processing."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            if image_path.lower().endswith('.pdf'):
                return self._process_pdf(image_path, max_pages)

            return self._process_image(image_path)

        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return []

    def _process_pdf(self, pdf_path, max_pages=None):
        """Process PDF with enhancement pipeline."""
        if not PDF_SUPPORT:
            logger.warning(f"PDF support unavailable. Skipping {pdf_path}")
            return []

        try:
            convert_kwargs = {
                'dpi': self.dpi,
                'fmt': 'jpeg',
                'thread_count': 1,
                'use_pdftocairo': True,
            }
            if max_pages:
                convert_kwargs['last_page'] = max_pages

            pages = convert_from_path(pdf_path, **convert_kwargs)

            temp_image_paths = []
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            import tempfile
            dir_name = tempfile.gettempdir()

            for i, page in enumerate(pages):
                try:
                    resized_page = self._resize_image_if_needed(page)
                    temp_path = os.path.join(dir_name, f"{base_name}_page_{i+1}.jpg")
                    resized_page.save(temp_path, 'JPEG', quality=self.JPEG_QUALITY)

                    if resized_page != page:
                        resized_page.close()
                    page.close()

                    # Apply enhancement pipeline
                    enhanced_path = self._enhance_image(temp_path)
                    if enhanced_path != temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)  # Remove unenhanced version

                    temp_image_paths.append(enhanced_path)

                except Exception as e:
                    logger.warning(f"Failed to process page {i+1}: {e}")

            del pages
            gc.collect()

            return temp_image_paths

        except Exception as e:
            logger.error(f"Failed to convert PDF {pdf_path}: {e}")
            return []

    def _process_image(self, image_path):
        """Process image file with enhancement."""
        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                width, height = img.size
                if width > self.max_size or height > self.max_size:
                    resized = self._resize_image_if_needed(img.copy())
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    import tempfile
                    dir_name = tempfile.gettempdir()
                    temp_path = os.path.join(dir_name, f"{base_name}_resized.jpg")
                    resized.save(temp_path, 'JPEG', quality=self.JPEG_QUALITY)
                    resized.close()

                    enhanced_path = self._enhance_image(temp_path)
                    if enhanced_path != temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
                    return [enhanced_path]

            # No resize needed — enhance in place (to temp file)
            enhanced_path = self._enhance_image(image_path)
            return [enhanced_path]

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return []

    @staticmethod
    def cleanup_temp_files(file_paths, original_path):
        """Remove temporary files created during processing."""
        for path in file_paths:
            if path != original_path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup {path}: {e}")
