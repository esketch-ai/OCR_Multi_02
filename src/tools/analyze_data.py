# -*- coding: utf-8 -*-
import os
import csv
import argparse
import sys
from dotenv import load_dotenv

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ocr.hybrid_engine import HybridOCREngine
from src.parser.car_registration import CarRegistrationParser
from src.ocr.preprocessor import ImagePreprocessor
from src.config import Config

def analyze_data(input_dirs, output_csv):
    print(f"Starting analysis on: {input_dirs}")
    
    load_dotenv()
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') is None:
         os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service_account.json'

    engine = HybridOCREngine()
    parser = CarRegistrationParser()
    preprocessor = ImagePreprocessor()
    
    results = []
    
    total_files = 0
    
    # Collect all files first
    files_to_process = []
    for d in input_dirs:
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith(('.jpg', '.png', '.jpeg', '.pdf')):
                        files_to_process.append(os.path.join(root, f))
    
    print(f"Found {len(files_to_process)} files to process.")
    
    # Initialize CSV with headers
    headers = [
        'filename', 'path', 'vehicle_no', 'vin', 'vehicle_type', 
        'model_name', 'vehicle_format', 'model_year', 'engine_type', 
        'owner_name', 'registration_date', 'vehicle_specs'
    ]
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
    
    for idx, filepath in enumerate(files_to_process):
        print(f"[{idx+1}/{len(files_to_process)}] Processing {filepath}...")
        try:
            images = preprocessor.load_image(filepath)
            if not images:
                print("  -> Failed to load image.")
                continue
            
            # Use first page only for now for analysis
            image_path = images[0]
            
            # Hybrid OCR
            res = engine.detect_text_hybrid(image_path)
            
            # Parse
            parsed = parser.parse_hybrid(res)
            
            # Record result
            row = {
                'filename': os.path.basename(filepath),
                'path': filepath,
                'vehicle_no': parsed.get('vehicle_no'),
                'vin': parsed.get('vin'),
                'vehicle_type': parsed.get('vehicle_type'),
                'model_name': parsed.get('model_name'),
                'vehicle_format': parsed.get('vehicle_format'),
                'model_year': parsed.get('model_year'),
                'engine_type': parsed.get('engine_type'),
                'owner_name': parsed.get('owner_name'),
                'registration_date': parsed.get('registration_date'),
                'vehicle_specs': parsed.get('vehicle_specs')
            }
            
            # Write immediately
            with open(output_csv, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerow(row)
            
            # Cleanup temp
            if filepath.lower().endswith('.pdf'):
                for p in images:
                    if os.path.exists(p): os.remove(p)
                    
        except Exception as e:
            print(f"  -> Error: {e}")
            
    print(f"\nAnalysis complete. Results saved to {output_csv}")

if __name__ == "__main__":
    parser_args = argparse.ArgumentParser()
    parser_args.add_argument('--dirs', nargs='+', default=['Data', 'processed', 'failed'], help='Directories to scan')
    parser_args.add_argument('--output', default='analysis_results.csv', help='Output CSV file')
    args = parser_args.parse_args()
    
    analyze_data(args.dirs, args.output)
