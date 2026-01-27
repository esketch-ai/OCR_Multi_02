
# -*- coding: utf-8 -*-
import os
import sys
from dotenv import load_dotenv

load_dotenv()
if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') is None:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service_account.json'

from src.ocr.hybrid_engine import HybridOCREngine
from src.ocr.preprocessor import ImagePreprocessor

def debug_files():
    print("Init Engine...")
    engine = HybridOCREngine()
    preprocessor = ImagePreprocessor()
    
    # Scan all files in processed to find the match
    print("Scanning processed directory...")
    
    count = 0
    for root, dirs, files in os.walk("processed"):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.png', '.pdf')):
                filepath = os.path.join(root, filename)
                count += 1
                if count % 20 == 0:
                    print(f"Scanned {count} files...")
                
                try:
                    images = preprocessor.load_image(filepath)
                    if not images: continue
                    
                    # Hybrid OCR
                    res = engine.detect_text_hybrid(images[0])
                    g_text = res['google']['text']
                    
                    # Match date loose
                    if "2020" in g_text and "11" in g_text and "14" in g_text:
                        print("\n>>> MATCH FOUND <<<")
                        print(f"File: {filepath}")
                        print("-" * 30)
                        print(g_text)
                        print("-" * 30)
                        return
                    
                    # Clean temp
                    if filepath.lower().endswith('.pdf'):
                        for p in images:
                            if os.path.exists(p): os.remove(p)
                except Exception as e:
                    print(f"Skipping {filename}: {e}")

if __name__ == "__main__":
    debug_files()
