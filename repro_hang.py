# -*- coding: utf-8 -*-
import os
import signal
import glob
from src.ocr.preprocessor import ImagePreprocessor
from src.ocr.hybrid_engine import HybridOCREngine
from dotenv import load_dotenv

load_dotenv()
if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') is None:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service_account.json'

def handler(signum, frame):
    raise TimeoutError("Timeout reached!")

# Register the signal function handler
signal.signal(signal.SIGALRM, handler)

def test_hang():
    # Use glob to find the file
    files = glob.glob("processed/*1016*.pdf")
    if not files:
        print("File not found via glob.")
        return
    
    f = files[0]
    print(f"Testing full process of {f}")
    
    pre = ImagePreprocessor()
    engine = HybridOCREngine()
    
    # Set 20s timeout
    signal.alarm(20)
    try:
        images = pre.load_image(f)
        print(f"Loaded {len(images)} images.")
        if images:
            print(f"Running OCR on {images[0]}")
            res = engine.detect_text_hybrid(images[0])
            print("OCR Completed.")
    except TimeoutError:
        print("Timeout reached during processing!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        signal.alarm(0)

if __name__ == "__main__":
    test_hang()
