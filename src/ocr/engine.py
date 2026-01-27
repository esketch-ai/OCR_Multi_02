
import io
from google.cloud import vision

class OCREngine:
    def __init__(self):
        try:
            self.client = vision.ImageAnnotatorClient()
        except Exception as e:
            print(f"Failed to initialize Vision Client: {e}")
            raise

    def detect_text(self, image_path):
        """
        Detects text in an image file using Google Cloud Vision API.
        Returns:
            tuple: (text, confidence_score)
            - text: The full detected text.
            - confidence_score: Average confidence of the blocks (0.0 to 1.0).
        """
        try:
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()

            image = vision.Image(content=content)

            # Use document_text_detection for dense text
            response = self.client.document_text_detection(image=image)
            
            if response.error.message:
                raise Exception(
                    '{}\nFor more info on error messages, check: '
                    'https://cloud.google.com/apis/design/errors'.format(
                        response.error.message))

            text = response.full_text_annotation.text
            
            # Calculate simple average confidence from pages->blocks
            total_conf = 0.0
            block_count = 0
            
            if response.full_text_annotation.pages:
                for page in response.full_text_annotation.pages:
                    for block in page.blocks:
                        # Blocks have confidence
                        total_conf += block.confidence
                        block_count += 1
            
            avg_confidence = (total_conf / block_count) if block_count > 0 else 0.0
            
            # Return text, average confidence, AND the raw response object for detailed analysis
            return text, avg_confidence, response.full_text_annotation

        except Exception as e:
            print(f"OCR Error: {e}")
            return None, 0.0
