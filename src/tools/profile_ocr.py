import time
import os
import sys
from dotenv import load_dotenv

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ocr.hybrid_engine import HybridOCREngine
from src.parser.car_registration import CarRegistrationParser
from src.ocr.preprocessor import ImagePreprocessor

def profile_ocr(filepath):
    print(f"Profiling: {filepath}")
    
    load_dotenv()
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') is None:
         os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service_account.json'

    t0 = time.time()
    engine = HybridOCREngine()
    print(f"Engine Init: {time.time() - t0:.2f}s")
    
    t1 = time.time()
    preprocessor = ImagePreprocessor()
    images = preprocessor.load_image(filepath)
    print(f"Image Load: {time.time() - t1:.2f}s")
    
    if not images:
        print("No images loaded")
        return

    image_path = images[0]
    
    t2 = time.time()
    res = engine.detect_text_hybrid(image_path)
    print(f"OCR Hybrid Detect: {time.time() - t2:.2f}s")
    
    t3 = time.time()
    parser = CarRegistrationParser()
    parsed = parser.parse_hybrid(res)
    print(f"Parsing: {time.time() - t3:.2f}s")
    
    print("-" * 20)
    print(parsed)
    
    # Cleanup
    if filepath.lower().endswith('.pdf'):
        for p in images:
            if os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    # Find a file
    target_file = None
    for root, dirs, files in os.walk("Data"):
        for f in files:
            if f.endswith(".pdf"):
                target_file = os.path.join(root, f)
                break
        if target_file: break
    
    if target_file:
        profile_ocr(target_file)
    else:
        print("No PDF found in Data")
