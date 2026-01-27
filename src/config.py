import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'service_account.json')
    GOOGLE_SHEET_ID = os.getenv('GOOGLE_SHEET_ID')

    # Input/Output directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_DIR = os.path.join(BASE_DIR, 'input')
    PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
    FAILED_DIR = os.path.join(BASE_DIR, 'failed')

    # OCR Settings
    OCR_LANGUAGE = 'ko'  # Korean

    # Memory Optimization Settings
    PDF_DPI = int(os.getenv('PDF_DPI', '150'))           # Lower DPI = less memory (default: 150)
    MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '2048'))  # Max dimension for images
    ENABLE_PADDLE = os.getenv('ENABLE_PADDLE', 'false').lower() == 'true'  # PaddleOCR off by default

    # Processing Settings
    GC_INTERVAL = int(os.getenv('GC_INTERVAL', '10'))    # Garbage collect every N files
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '50'))      # Files per batch
    RATE_LIMIT_SECONDS = float(os.getenv('RATE_LIMIT', '1.5'))  # Seconds between API calls

    @classmethod
    def validate(cls):
        if not cls.GOOGLE_SHEET_ID:
            print("Warning: GOOGLE_SHEET_ID not found in .env")
        if not os.path.exists(cls.GOOGLE_APPLICATION_CREDENTIALS):
            print(f"Warning: Service account file not found at {cls.GOOGLE_APPLICATION_CREDENTIALS}")


# Validate on import
Config.validate()
