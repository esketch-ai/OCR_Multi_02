# -*- coding: utf-8 -*-
import os
import sys
import time
import multiprocessing
import traceback
from dotenv import load_dotenv

# Worker function must be top-level for pickling
def process_single_file(filepath):
    """
    Worker function to process a single file.
    Initializes engine completely fresh to ensure no shared state issues.
    """
    try:
        # Lazy import to avoid loading heavy libs in main process
        from src.ocr.hybrid_engine import HybridOCREngine
        from src.ocr.preprocessor import ImagePreprocessor
        
        # Load env vars inside worker if needed (though they are inherited)
        # load_dotenv() 

        preprocessor = ImagePreprocessor()
        # Initialize Engine (heavy load)
        engine = HybridOCREngine()
        
        print(f"[Worker] Processing: {filepath}")
        
        images = preprocessor.load_image(filepath)
        if not images:
            return None
        
        result_text = ""
        # Process first page/image
        if len(images) > 0:
            res = engine.detect_text_hybrid(images[0])
            result_text = res.get('google', {}).get('text', '')
            
            # Cleanup temp files immediately
            if filepath.lower().endswith('.pdf'):
                for p in images:
                    if os.path.exists(p): 
                        try: os.remove(p)
                        except: pass
        
        return {
            'filepath': filepath,
            'text': result_text,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"[Worker] Error processing {filepath}: {e}")
        # traceback.print_exc()
        return {'filepath': filepath, 'status': 'error', 'error': str(e)}

def debug_files_robust():
    # Load environment variables
    load_dotenv()
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') is None:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service_account.json'

    print("Scanning processed directory...")
    target_files = []
    
    # "»óÁÖ" -> \uc0c1\uc8fc
    target_owner_part = "\uc0c1\uc8fc"
    
    # Simple bad file filter
    SKIP_PARTIALS = ["1016"] 

    for root, dirs, files in os.walk("processed"):
        for filename in files:
            if any(s in filename for s in SKIP_PARTIALS):
                print(f"Skipping known bad file: {filename}")
                continue
                
            if filename.lower().endswith(('.jpg', '.png', '.pdf')):
                filepath = os.path.join(root, filename)
                target_files.append(filepath)
    
    print(f"Found {len(target_files)} files to process.")
    
    # Use a safe number of processes. 
    # Since OCR uses high CPU/RAM, stick to 1 or 2 processes.
    # maxtasksperchild=5 ensures we kill the worker often to free memory.
    with multiprocessing.Pool(processes=1, maxtasksperchild=5) as pool:
        
        # Use imap_unordered to get results as they finish
        # We can also wrap with timeout logic if needed, but Pool handles basics.
        
        results = pool.imap_unordered(process_single_file, target_files)
        
        try:
            for res in results:
                if not res: continue
                
                if res['status'] == 'error':
                    # Error logged in worker
                    continue
                
                filepath = res['filepath']
                g_text = res['text']
                
                # Check Logic
                # Match date "2011-11-21" -> "2011" and "11" and "21"
                if g_text and "2011" in g_text and "11" in g_text and "21" in g_text:
                    # Check owner
                    if target_owner_part in g_text:
                        print("\n>>> MATCH FOUND <<<")
                        print(f"File: {filepath}")
                        print("-" * 30)
                        print(g_text)
                        print("-" * 30)
                        # We could stop here, but pool is running. 
                        # To stop strictly: pool.terminate() return
                        # But maybe we want to find ALL matches?
                        # Let's run all.
        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt, terminating pool...")
            pool.terminate()
            pool.join()
        except Exception as e:
            print(f"Main loop error: {e}")
            pool.terminate()

if __name__ == "__main__":
    # Ensure safe multiprocessing on macOS (spawn)
    multiprocessing.set_start_method('spawn', force=True)
    debug_files_robust()
