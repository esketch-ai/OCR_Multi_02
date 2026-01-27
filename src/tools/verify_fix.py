
import sys
import os
import glob
from src.ocr.engine import OCREngine
from src.ocr.preprocessor import ImagePreprocessor
from src.parser.car_registration import CarRegistrationParser
from src.config import Config

def verify_fix(input_dir):
    print(f"Verifying field extraction on files in {input_dir}")
    
    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = Config.GOOGLE_APPLICATION_CREDENTIALS
    
    ocr_engine = OCREngine()
    parser = CarRegistrationParser()
    preprocessor = ImagePreprocessor()
    
    files = glob.glob(os.path.join(input_dir, "**/*.pdf"), recursive=True)[:2]
        
    for file_path in files:
        print(f"\nProcessing: {os.path.basename(file_path)}")
        image_paths = preprocessor.load_image(file_path)
        
        if not image_paths:
            continue
            
        text, conf = ocr_engine.detect_text(image_paths[0])
        parsed = parser.parse(text)
        
        print(f"No: {parsed.get('vehicle_no')}")
        print(f"Type: {parsed.get('vehicle_type')}")
        print(f"Name: {parsed.get('model_name')}")
        print(f"Format: {parsed.get('vehicle_format')}")
        print(f"Year: {parsed.get('model_year')}")
        print(f"Engine: {parsed.get('engine_type')}")
        
        if file_path.lower().endswith('.pdf') and os.path.exists(image_paths[0]):
            os.remove(image_paths[0])

if __name__ == "__main__":
    verify_fix(sys.argv[1] if len(sys.argv)>1 else "Data")
