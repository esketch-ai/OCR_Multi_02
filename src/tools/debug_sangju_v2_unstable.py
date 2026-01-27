# -*- coding: utf-8 -*-
import os
import sys
import gc
import time
import psutil
from dotenv import load_dotenv

load_dotenv()
if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') is None:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service_account.json'

from src.ocr.hybrid_engine import HybridOCREngine
from src.ocr.preprocessor import ImagePreprocessor

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[Memory] RSS: {mem_info.rss / 1024 / 1024:.2f} MB")

def debug_files():
    print("Init Engine...")
    print_memory_usage()
    
    engine = HybridOCREngine()
    preprocessor = ImagePreprocessor()
    
    print("Scanning processed directory for Sangju matches...")
    count = 0
    
    # "»óÁÖ" -> \uc0c1\uc8fc
    target_owner_part = "\uc0c1\uc8fc" 
    
    # Files known to cause hangs/crashes
    SKIP_FILES = [
        "1016" # Match part of filename to avoid encoding issues
    ]
    
    for root, dirs, files in os.walk("processed"):
        for filename in files:
            # Skip known bad files
            if any(skip in filename for skip in SKIP_FILES):
                print(f"Skipping known bad file: {filename}")
                continue

            if filename.lower().endswith(('.jpg', '.png', '.pdf')):
                filepath = os.path.join(root, filename)
                count += 1
                
                # Monitor and Cleanup every 10 files
                if count % 10 == 0:
                    print(f"Scanned {count} files...")
                    print_memory_usage()
                    gc.collect()
                
                print(f"Processing: {filepath}")
                
                try:
                    images = preprocessor.load_image(filepath)
                    if not images: continue
                    
                    # Hybrid OCR
                    if len(images) > 0:
                        res = engine.detect_text_hybrid(images[0])
                        g_text = res['google']['text']
                        
                        # Match date "2011-11-21" -> "2011" and "11" and "21"
                        if g_text and "2011" in g_text and "11" in g_text and "21" in g_text:
                            # Check owner
                            if target_owner_part in g_text:
                                print("\n>>> MATCH FOUND <<<")
                                print(f"File: {filepath}")
                                print("-" * 30)
                                print(g_text)
                                print("-" * 30)
                                
                                # Cleanup before return
                                if filepath.lower().endswith('.pdf'):
                                    for p in images:
                                        if os.path.exists(p): os.remove(p)
                                del images
                                del res
                                return
                        
                        del res
                        del g_text

                    # Cleanup temp
                    if filepath.lower().endswith('.pdf'):
                        for p in images:
                            if os.path.exists(p): os.remove(p)
                    
                    del images
                    
                except Exception as e:
                    print(f"Skipping {filename}: {e}")
                
                # Optional: Sleep briefly to yield resources
                # time.sleep(0.05)

if __name__ == "__main__":
    debug_files()
