
# -*- coding: utf-8 -*-
import os
import sys
from dotenv import load_dotenv

# Load env before imports that might need it
load_dotenv()
# Set creds explicitly if needed
if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') is None:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service_account.json'

from src.ocr.hybrid_engine import HybridOCREngine
from src.ocr.preprocessor import ImagePreprocessor

def find_target_file(search_dir):
    print("Initializing components...")
    engine = HybridOCREngine()
    preprocessor = ImagePreprocessor()
    
    target_date = "2020-11-14"
    
    print(f"Walking directory: {search_dir}")
    
    count = 0
    for root, dirs, files in os.walk(search_dir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.png', '.pdf')):
                count += 1
                filepath = os.path.join(root, filename)
                
                # We can optimize by only checking files in directories that look relevant if needed
                # But to be safe from encoding match fail, we check content.
                
                try:
                    # Load image
                    image_paths = preprocessor.load_image(filepath)
                    if not image_paths: continue
                    
                    # Check first page content
                    # Use Google Engine wrapper from Hybrid for speed? 
                    # detect_text_hybrid does both.
                    
                    hybrid_res = engine.detect_text_hybrid(image_paths[0])
                    g_text = hybrid_res['google']['text']
                    
                    if target_date in g_text:
                        print(f"\n[MATCH FOUND] File: {filename}")
                        print(f"Path: {filepath}")
                        print("-" * 20)
                        # Print ASCII safe preview?
                        print("Text length:", len(g_text))
                        print("-" * 20)
                        return filepath
                        
                    # Cleanup
                    if filepath.lower().endswith('.pdf'):
                        for p in image_paths:
                            if os.path.exists(p): os.remove(p)
                            
                except Exception as e:
                    print(f"Error scanning {filename}: {e}")
                    
    print(f"Scanned {count} files. No match found.")

if __name__ == "__main__":
    find_target_file("processed")
