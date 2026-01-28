import os
import gc
from PIL import Image

try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


class ImagePreprocessor:
    """
    Image preprocessor with memory optimization.
    """
    # Default settings for memory efficiency
    DEFAULT_DPI = 150          # Reduced from 300 for memory efficiency
    MAX_IMAGE_SIZE = 2048      # Maximum dimension for OCR processing
    JPEG_QUALITY = 85          # Quality for JPEG compression

    def __init__(self, dpi=None, max_size=None):
        self.dpi = dpi or self.DEFAULT_DPI
        self.max_size = max_size or self.MAX_IMAGE_SIZE

    def _resize_image_if_needed(self, img):
        """
        Resize image if it exceeds maximum dimension while maintaining aspect ratio.
        Returns resized PIL Image.
        """
        width, height = img.size
        if width <= self.max_size and height <= self.max_size:
            return img

        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = self.max_size
            new_height = int(height * (self.max_size / width))
        else:
            new_height = self.max_size
            new_width = int(width * (self.max_size / height))

        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def load_image(self, image_path, max_pages=None):
        """
        Loads an image or PDF with memory optimization.
        Returns a LIST of image paths (temporary paths for PDF pages) or the original path for images.
        :param max_pages: If set, limits the number of pages converted from PDF.
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Handle PDF
            if image_path.lower().endswith('.pdf'):
                return self._process_pdf(image_path, max_pages)

            # Handle Image
            return self._process_image(image_path)

        except Exception as e:
            print(f"Error loading image: {e}")
            return []

    def _process_pdf(self, pdf_path, max_pages=None):
        """Process PDF file with memory-optimized settings."""
        if not PDF_SUPPORT:
            print(f"Warning: PDF support unavailable (install pdf2image and poppler). Skipping {pdf_path}")
            return []

        try:
            # Memory-optimized PDF conversion settings
            convert_kwargs = {
                'dpi': self.dpi,
                'fmt': 'jpeg',           # JPEG uses less memory than PNG
                'thread_count': 1,       # Limit threads to reduce memory
                'use_pdftocairo': True,  # More memory efficient
            }
            if max_pages:
                convert_kwargs['last_page'] = max_pages

            pages = convert_from_path(pdf_path, **convert_kwargs)

            temp_image_paths = []
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            # Use system temp directory instead of source directory
            import tempfile
            dir_name = tempfile.gettempdir()

            # Save each page as a temp image with resizing
            for i, page in enumerate(pages):
                try:
                    # Resize if needed
                    resized_page = self._resize_image_if_needed(page)

                    # Save as JPEG for memory efficiency
                    temp_path = os.path.join(dir_name, f"{base_name}_page_{i+1}.jpg")
                    resized_page.save(temp_path, 'JPEG', quality=self.JPEG_QUALITY)

                    # Cleanup PIL image immediately
                    if resized_page != page:
                        resized_page.close()
                    page.close()

                    # Verify the saved image
                    with Image.open(temp_path) as verify_img:
                        verify_img.verify()
                    temp_image_paths.append(temp_path)

                except Exception as e:
                    print(f"Warning: Failed to process page {i+1}. Error: {e}")
                    if 'temp_path' in locals() and os.path.exists(temp_path):
                        os.remove(temp_path)

            # Force garbage collection after PDF processing
            del pages
            gc.collect()

            return temp_image_paths

        except Exception as e:
            print(f"Failed to convert PDF {pdf_path}: {e}")
            return []

    def _process_image(self, image_path):
        """Process image file with optional resizing."""
        try:
            with Image.open(image_path) as img:
                img.verify()

            # Check if resizing is needed
            with Image.open(image_path) as img:
                width, height = img.size
                if width > self.max_size or height > self.max_size:
                    # Need to resize - create temp file in system temp directory
                    resized = self._resize_image_if_needed(img.copy())
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    import tempfile
                    dir_name = tempfile.gettempdir()
                    temp_path = os.path.join(dir_name, f"{base_name}_resized.jpg")
                    resized.save(temp_path, 'JPEG', quality=self.JPEG_QUALITY)
                    resized.close()
                    return [temp_path]

            return [image_path]  # Return as list for consistent handling

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return []

    @staticmethod
    def cleanup_temp_files(file_paths, original_path):
        """
        Remove temporary files created during processing.
        Won't remove the original file.
        """
        for path in file_paths:
            if path != original_path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Failed to cleanup temp file {path}: {e}")
