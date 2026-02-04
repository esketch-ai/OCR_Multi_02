# -*- coding: utf-8 -*-
"""
Fast parallel processing of vehicle registration certificates using LOCAL OCR.
Uses multiprocessing for parallel OCR processing.

Features:
- Recursive directory scanning
- Archive extraction (zip, tar, tar.gz, rar, 7z)
- Multi-page PDF support
- Parallel processing
"""
import os
import sys
import time
import shutil
import argparse
import unicodedata
import logging
import warnings
import zipfile
import tarfile
import tempfile
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count, set_start_method

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Supported file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
PDF_EXTENSIONS = {'.pdf'}
ARCHIVE_EXTENSIONS = {'.zip', '.tar', '.tar.gz', '.tgz', '.gz', '.rar', '.7z'}


def normalize_path(path):
    """Normalize Unicode path for consistent handling."""
    return unicodedata.normalize('NFC', path)


def is_image_file(path):
    """Check if file is a supported image."""
    return os.path.splitext(path.lower())[1] in IMAGE_EXTENSIONS


def is_pdf_file(path):
    """Check if file is a PDF."""
    return os.path.splitext(path.lower())[1] in PDF_EXTENSIONS


def is_archive_file(path):
    """Check if file is a supported archive."""
    lower = path.lower()
    return any(lower.endswith(ext) for ext in ARCHIVE_EXTENSIONS)


def is_processable_file(path):
    """Check if file can be processed (image or PDF)."""
    return is_image_file(path) or is_pdf_file(path)


def extract_archive(archive_path, extract_to):
    """
    Extract archive to specified directory.

    Supports: zip, tar, tar.gz, tgz
    Returns list of extracted file paths.
    """
    extracted_files = []
    archive_path = normalize_path(archive_path)
    lower = archive_path.lower()

    try:
        if lower.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                # Handle Korean filename encoding
                for info in zf.infolist():
                    try:
                        # Try to decode as CP437 (common for Korean Windows zip)
                        decoded_name = info.filename.encode('cp437').decode('euc-kr')
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        try:
                            decoded_name = info.filename.encode('cp437').decode('utf-8')
                        except:
                            decoded_name = info.filename

                    # Normalize the path
                    decoded_name = normalize_path(decoded_name)

                    # Extract
                    target_path = os.path.join(extract_to, decoded_name)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)

                    if not info.is_dir():
                        with zf.open(info) as source:
                            with open(target_path, 'wb') as target:
                                target.write(source.read())
                        extracted_files.append(target_path)

        elif lower.endswith(('.tar', '.tar.gz', '.tgz', '.gz')):
            mode = 'r:gz' if lower.endswith(('.tar.gz', '.tgz', '.gz')) else 'r'
            with tarfile.open(archive_path, mode) as tf:
                tf.extractall(extract_to)
                for member in tf.getmembers():
                    if member.isfile():
                        extracted_files.append(os.path.join(extract_to, member.name))

        elif lower.endswith('.rar'):
            # Try to use unrar command
            import subprocess
            result = subprocess.run(['unrar', 'x', '-y', archive_path, extract_to],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for root, dirs, files in os.walk(extract_to):
                    for f in files:
                        extracted_files.append(os.path.join(root, f))
            else:
                logger.warning(f"Failed to extract RAR: {archive_path}")

        elif lower.endswith('.7z'):
            # Try to use 7z command
            import subprocess
            result = subprocess.run(['7z', 'x', '-y', f'-o{extract_to}', archive_path],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for root, dirs, files in os.walk(extract_to):
                    for f in files:
                        extracted_files.append(os.path.join(root, f))
            else:
                logger.warning(f"Failed to extract 7z: {archive_path}")

    except Exception as e:
        logger.error(f"Error extracting {archive_path}: {e}")

    return [normalize_path(f) for f in extracted_files]


def scan_directory(root_dir, extract_archives=True, temp_dir=None):
    """
    Recursively scan directory for processable files.

    Args:
        root_dir: Directory to scan
        extract_archives: If True, extract and scan archive contents
        temp_dir: Directory for extracted archives

    Returns:
        List of file paths to process
    """
    files_to_process = []
    root_dir = normalize_path(root_dir)

    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix='ocr_extract_')

    logger.info(f"Scanning directory: {root_dir}")

    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for filename in files:
            if filename.startswith('.'):
                continue

            filepath = normalize_path(os.path.join(root, filename))

            if is_processable_file(filepath):
                files_to_process.append(filepath)

            elif extract_archives and is_archive_file(filepath):
                # Extract archive and scan contents
                archive_name = os.path.splitext(os.path.basename(filepath))[0]
                extract_subdir = os.path.join(temp_dir, archive_name)
                os.makedirs(extract_subdir, exist_ok=True)

                logger.info(f"Extracting archive: {filename}")
                extracted = extract_archive(filepath, extract_subdir)

                for extracted_file in extracted:
                    if is_processable_file(extracted_file):
                        files_to_process.append(extracted_file)
                    elif is_archive_file(extracted_file):
                        # Nested archive
                        nested_files = scan_directory(
                            os.path.dirname(extracted_file),
                            extract_archives=True,
                            temp_dir=temp_dir
                        )
                        files_to_process.extend(nested_files)

    return files_to_process


def get_pdf_page_count(pdf_path):
    """Get number of pages in PDF."""
    try:
        from pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(pdf_path)
        return info.get('Pages', 1)
    except:
        return 1


def sanitize_filename(text):
    """Remove invalid characters from filename."""
    if not text:
        return ''
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        text = text.replace(char, '_')
    return text.strip()


def init_worker():
    """Initialize OCR engine in worker process."""
    import warnings
    import os
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['GLOG_minloglevel'] = '3'
    # Suppress PaddleOCR logging
    import logging
    logging.getLogger('ppocr').setLevel(logging.ERROR)


def process_single_file(args):
    """
    Process a single file. Called by worker processes.

    Args:
        args: tuple of (filepath, output_dir, dpi, use_structured)

    Returns:
        dict with processing results
    """
    import os
    import warnings
    import logging as log

    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['GLOG_minloglevel'] = '3'
    log.getLogger('ppocr').setLevel(log.ERROR)

    # Unpack args - support both old (3) and new (4) formats
    if len(args) == 4:
        filepath, output_dir, dpi, use_structured = args
    else:
        filepath, output_dir, dpi = args
        use_structured = True  # Default to structured parser

    filename = os.path.basename(filepath)

    try:
        from paddleocr import PaddleOCR
        from PIL import Image
        import tempfile

        # Initialize OCR for this worker (lazy initialization)
        # Balanced configuration: mobile detection + Korean recognition
        # Higher det_limit for better accuracy on registration certificates
        ocr = PaddleOCR(
            lang='korean',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name='PP-OCRv5_mobile_det',
            text_recognition_model_name='korean_PP-OCRv5_mobile_rec',
            text_det_limit_side_len=1920,  # Higher for better accuracy
            text_det_limit_type='max',
            text_det_box_thresh=0.5,       # Lower threshold to catch more text
        )

        # Import parsers
        from src.parser.car_registration import CarRegistrationParser
        from src.parser.structured_parser import StructuredRegistrationParser
        from src.parser.template_extractor import TemplateExtractor
        from src.validator.vin_validator import VINValidator

        legacy_parser = CarRegistrationParser()
        structured_parser = StructuredRegistrationParser()
        template_extractor = TemplateExtractor()
        vin_validator = VINValidator()

        # Check if file exists
        if not os.path.exists(filepath):
            return {'status': 'skip', 'reason': 'file_not_found', 'filename': filename}

        # Helper: sanitize filename for temp paths (remove special chars)
        import re
        import uuid
        safe_basename = re.sub(r'[^\w\-.]', '_', os.path.splitext(filename)[0])
        unique_id = uuid.uuid4().hex[:8]

        # Handle PDF vs Image
        if filepath.lower().endswith('.pdf'):
            from pdf2image import convert_from_path
            pages = convert_from_path(filepath, dpi=dpi, first_page=1, last_page=1)
            if not pages:
                return {'status': 'skip', 'reason': 'pdf_empty', 'filename': filename}

            # Save to temp file with sanitized name
            temp_path = os.path.join(tempfile.gettempdir(), f"{safe_basename}_{unique_id}_p1.jpg")
            # Convert to RGB if needed (PDF pages are usually RGB already)
            page_img = pages[0]
            if page_img.mode in ('RGBA', 'LA', 'P'):
                page_img = page_img.convert('RGB')
            page_img.save(temp_path, 'JPEG', quality=85)
            pages[0].close()
            image_path = temp_path
        else:
            # Check if image needs resizing or conversion
            with Image.open(filepath) as img:
                width, height = img.size
                needs_conversion = img.mode in ('RGBA', 'LA', 'P')
                needs_resize = width > 2048 or height > 2048

                if needs_resize or needs_conversion:
                    # Convert RGBA/LA/P to RGB first
                    if needs_conversion:
                        img = img.convert('RGB')

                    # Resize if needed
                    if needs_resize:
                        if width > height:
                            new_width = 2048
                            new_height = int(height * (2048 / width))
                        else:
                            new_height = 2048
                            new_width = int(width * (2048 / height))
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    temp_path = os.path.join(tempfile.gettempdir(), f"{safe_basename}_{unique_id}_proc.jpg")
                    img.save(temp_path, 'JPEG', quality=85)
                    image_path = temp_path
                else:
                    image_path = filepath

        # Run OCR
        result = ocr.predict(image_path)
        res = result[0].json['res']
        texts = res.get('rec_texts', [])
        scores = res.get('rec_scores', [])

        if not texts:
            return {'status': 'skip', 'reason': 'no_text', 'filename': filename}

        full_text = '\n'.join(texts)

        # Verify document type
        if not structured_parser.verify_document_type(full_text):
            return {'status': 'skip', 'reason': 'not_registration', 'filename': filename}

        # Create raw result format with bounding boxes
        raw_result = []
        boxes = res.get('rec_polys', [])
        for i, (text, score) in enumerate(zip(texts, scores)):
            box = boxes[i] if i < len(boxes) else [[0, 0], [0, 0], [0, 0], [0, 0]]
            raw_result.append([box, (text, score)])

        # Create hybrid result format for parser compatibility
        hybrid_result = {
            'google': {
                'text': full_text,
                'avg_conf': sum(scores) / len(scores) if scores else 0.0,
                'annotation': None
            },
            'paddle': {
                'result': raw_result,
                'text_lines': texts
            }
        }

        # Primary: Use legacy regex-based parser (more reliable for Korean documents)
        parsed_data = legacy_parser.parse_hybrid(hybrid_result, filename=filename)
        confidences = parsed_data.get('confidences', {})

        # Enhanced mode: Use multiple extraction methods to supplement
        if use_structured:
            # Method 1: Pattern-based extraction using ①-㉒ markers
            pattern_data = structured_parser.extract_by_circled_patterns(full_text)
            pattern_conf = pattern_data.pop('confidences', {})

            # Method 2: Template-based extraction using position
            from PIL import Image
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            template_data = template_extractor.extract_from_boxes(raw_result, img_width, img_height)
            template_conf = template_data.pop('confidences', {})

            # All fields to supplement
            supplement_fields = [
                # Main registration fields (①-⑩)
                'vehicle_no', 'vehicle_type', 'model_name', 'vehicle_format',
                'model_year', 'vin', 'engine_type', 'owner_name', 'owner_address',
                # Specification fields (제원 ①-⑫)
                'spec_no', 'length_mm', 'width_mm', 'height_mm', 'total_weight_kg',
                'displacement_cc', 'rated_output', 'passenger_capacity', 'max_load_kg',
                'cylinders', 'fuel_type', 'type_approval_no',
                # Additional fields
                'first_registration_date', 'usage_type'
            ]

            # Fill from pattern extraction first
            for field in supplement_fields:
                if not parsed_data.get(field) and pattern_data.get(field):
                    struct_val = pattern_data[field]
                    if struct_val and len(str(struct_val)) >= 1:
                        parsed_data[field] = struct_val
                        confidences[field] = pattern_conf.get(field, 0.0)

            # Fill from template extraction for remaining missing fields
            for field in supplement_fields:
                if not parsed_data.get(field) and template_data.get(field):
                    tmpl_val = template_data[field]
                    if tmpl_val and len(str(tmpl_val)) >= 1:
                        parsed_data[field] = tmpl_val
                        confidences[field] = template_conf.get(field, 0.0)

        # Validate VIN
        vin = parsed_data.get('vin')
        vehicle_no = parsed_data.get('vehicle_no')
        is_valid, validation_msg = vin_validator.validate(vin)

        fuel_type = parsed_data.get('fuel_type', 'Unknown')

        # Copy file to output
        copied_path = None
        if vehicle_no and output_dir:
            vehicle_no_clean = sanitize_filename(vehicle_no)
            fuel_type_clean = sanitize_filename(fuel_type) if fuel_type else 'Unknown'
            _, ext = os.path.splitext(filepath)
            new_filename = f"{vehicle_no_clean}_{fuel_type_clean}{ext}"

            os.makedirs(output_dir, exist_ok=True)
            dest_path = os.path.join(output_dir, new_filename)

            if os.path.exists(dest_path):
                timestamp = datetime.now().strftime('%H%M%S')
                new_filename = f"{vehicle_no_clean}_{fuel_type_clean}_{timestamp}{ext}"
                dest_path = os.path.join(output_dir, new_filename)

            shutil.copy2(filepath, dest_path)
            copied_path = new_filename

        # Cleanup temp files
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            'status': 'success',
            'filename': filename,
            # Main registration fields (①-⑩)
            'vehicle_no': vehicle_no,                                    # ①
            'vehicle_type': parsed_data.get('vehicle_type', ''),         # ②
            'model_name': parsed_data.get('model_name', ''),             # ③
            'vehicle_format': parsed_data.get('vehicle_format', ''),     # ④
            'model_year': parsed_data.get('model_year', ''),             # ⑤
            'vin': vin,                                                  # ⑥
            'vin_valid': is_valid,
            'engine_type': parsed_data.get('engine_type', ''),           # ⑦
            'owner_name': parsed_data.get('owner_name', ''),             # ⑧
            'owner_address': parsed_data.get('owner_address', ''),       # ⑩
            # Specification fields (제원 ①-⑫)
            'length_mm': parsed_data.get('length_mm', ''),               # 제원②
            'width_mm': parsed_data.get('width_mm', ''),                 # 제원③
            'height_mm': parsed_data.get('height_mm', ''),               # 제원④
            'total_weight_kg': parsed_data.get('total_weight_kg', ''),   # 제원⑤
            'displacement_cc': parsed_data.get('displacement_cc', ''),   # 제원⑥
            'passenger_capacity': parsed_data.get('passenger_capacity', ''),  # 제원⑧
            'fuel_type': fuel_type,                                      # 제원⑪
            # Additional
            'first_registration_date': parsed_data.get('first_registration_date',
                                       parsed_data.get('registration_date', '')),
            'usage_type': parsed_data.get('usage_type', ''),
            'copied_file': copied_path,
            'conf_veh': confidences.get('vehicle_no', 0.0),
            'conf_vin': confidences.get('vin', 0.0),
        }

    except Exception as e:
        return {'status': 'error', 'filename': filename, 'error': str(e)}


def process_files_parallel(file_list_path, output_dir, start_idx=0, max_files=None,
                           num_workers=None, dpi=100, use_structured=True):
    """
    Process files in parallel using multiprocessing.

    Args:
        use_structured: If True, uses position-based extraction focusing on
                       circled numbers ①-㉒ for improved accuracy
    """
    # Read file list
    with open(file_list_path, 'r', encoding='utf-8') as f:
        all_files = [normalize_path(line.strip()) for line in f if line.strip()]

    # Apply start index and max files
    files_to_process = all_files[start_idx:]
    if max_files:
        files_to_process = files_to_process[:max_files]

    total_files = len(files_to_process)

    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 4)  # Limit to 4 for memory reasons

    logger.info(f"Processing {total_files} files with {num_workers} workers")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"DPI: {dpi}")
    logger.info(f"Parser mode: {'Structured (①-㉒)' if use_structured else 'Legacy'}")

    # Prepare arguments for each file (filepath, output_dir, dpi, use_structured)
    args_list = [(fp, output_dir, dpi, use_structured) for fp in files_to_process]

    # Initialize CSV file for results
    import csv
    csv_path = os.path.join(output_dir, f"ocr_results_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    os.makedirs(output_dir, exist_ok=True)

    # CSV header with all 22+ fields from ①-㉒ extraction
    header = [
        # Basic
        "파일명",
        # Main registration (①-⑩)
        "①차량번호", "②차종", "③차명", "④형식", "⑤모델연도",
        "⑥차대번호(VIN)", "VIN유효성", "⑦원동기형식", "⑧소유자", "⑩사용본거지",
        # Specifications (제원 ⑫-⑳)
        "⑫길이(mm)", "⑬너비(mm)", "⑭높이(mm)", "⑮총중량(kg)", "⑯배기량(cc)", "⑱승차정원",
        # Fuel type and additional
        "㉒연료타입", "최초등록일", "용도",
        # Meta
        "복사파일", "처리시각"
    ]

    # Create CSV file with header
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    logger.info(f"CSV output: {csv_path}")

    # Process in parallel
    start_time = time.time()
    success_count = 0
    skip_count = 0
    error_count = 0

    # Use spawn to avoid fork issues with PaddleOCR
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    logger.info("Starting parallel processing...")

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results = pool.imap_unordered(process_single_file, args_list, chunksize=1)

        for i, result in enumerate(results):
            status = result.get('status')
            filename = result.get('filename', 'unknown')

            if status == 'success':
                success_count += 1
                logger.info(f"[{i+1}/{total_files}] ✓ {filename} -> {result.get('vehicle_no')}, {result.get('vin')}")

                # Append to CSV with all fields
                row = [
                    filename,
                    # Main registration (①-⑩)
                    result.get('vehicle_no', ''),
                    result.get('vehicle_type', ''),
                    result.get('model_name', ''),
                    result.get('vehicle_format', ''),
                    result.get('model_year', ''),
                    result.get('vin', ''),
                    "Valid" if result.get('vin_valid') else "Invalid",
                    result.get('engine_type', ''),
                    result.get('owner_name', ''),
                    result.get('owner_address', ''),
                    # Specifications
                    result.get('length_mm', ''),
                    result.get('width_mm', ''),
                    result.get('height_mm', ''),
                    result.get('total_weight_kg', ''),
                    result.get('displacement_cc', ''),
                    result.get('passenger_capacity', ''),
                    # Additional
                    result.get('fuel_type', ''),
                    result.get('first_registration_date', ''),
                    result.get('usage_type', ''),
                    # Meta
                    result.get('copied_file', ''),
                    time.strftime("%Y-%m-%d %H:%M:%S")
                ]
                with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

            elif status == 'skip':
                skip_count += 1
                logger.info(f"[{i+1}/{total_files}] - {filename}: SKIP ({result.get('reason')})")
            else:
                error_count += 1
                logger.warning(f"[{i+1}/{total_files}] ✗ {filename}: {result.get('error')}")

    elapsed = time.time() - start_time
    avg_time = elapsed / total_files if total_files > 0 else 0

    logger.info(f"\n{'='*50}")
    logger.info(f"Processing complete!")
    logger.info(f"  Total files: {total_files}")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Skipped: {skip_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Total time: {elapsed:.1f}s")
    logger.info(f"  Avg time per file: {avg_time:.1f}s")
    logger.info(f"  CSV output: {csv_path}")
    logger.info(f"{'='*50}")


def expand_multipage_pdfs(file_list, dpi=150):
    """
    Expand multi-page PDFs into separate entries.
    Each page becomes a separate processing task.

    Returns list of tuples: (filepath, page_number) where page_number is None for images.
    """
    expanded_list = []

    for filepath in file_list:
        if is_pdf_file(filepath):
            try:
                page_count = get_pdf_page_count(filepath)
                if page_count > 1:
                    logger.info(f"Multi-page PDF ({page_count} pages): {os.path.basename(filepath)}")
                    for page in range(1, page_count + 1):
                        expanded_list.append((filepath, page))
                else:
                    expanded_list.append((filepath, 1))
            except Exception as e:
                logger.warning(f"Error checking PDF pages: {filepath}: {e}")
                expanded_list.append((filepath, 1))
        else:
            expanded_list.append((filepath, None))

    return expanded_list


def process_files_from_list(files, output_dir, start_idx=0, max_files=None,
                            num_workers=None, dpi=150, use_structured=True):
    """
    Process files from a list (supports multi-page PDFs).
    """
    # Expand multi-page PDFs
    expanded_files = expand_multipage_pdfs(files, dpi)

    # Apply start index and max files
    files_to_process = expanded_files[start_idx:]
    if max_files:
        files_to_process = files_to_process[:max_files]

    total_files = len(files_to_process)

    if total_files == 0:
        logger.warning("No files to process!")
        return

    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 4)

    logger.info(f"Processing {total_files} items with {num_workers} workers")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"DPI: {dpi}")
    logger.info(f"Parser mode: {'Structured (①-㉒)' if use_structured else 'Legacy'}")

    # Prepare arguments - include page number for PDFs
    args_list = [(fp, page, output_dir, dpi, use_structured) for fp, page in files_to_process]

    # Initialize CSV file
    import csv
    csv_path = os.path.join(output_dir, f"ocr_results_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    os.makedirs(output_dir, exist_ok=True)

    # CSV header
    header = [
        "파일명", "페이지",
        "①차량번호", "②차종", "③차명", "④형식", "⑤모델연도",
        "⑥차대번호(VIN)", "VIN유효성", "⑦원동기형식", "⑧소유자", "⑩사용본거지",
        "⑫길이(mm)", "⑬너비(mm)", "⑭높이(mm)", "⑮총중량(kg)", "⑯배기량(cc)", "⑱승차정원",
        "㉒연료타입", "최초등록일", "용도",
        "원본경로", "처리시각"
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    logger.info(f"CSV output: {csv_path}")

    # Process in parallel
    start_time = time.time()
    success_count = 0
    skip_count = 0
    error_count = 0

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    logger.info("Starting parallel processing...")

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results = pool.imap_unordered(process_single_file_v2, args_list, chunksize=1)

        for i, result in enumerate(results):
            status = result.get('status')
            filename = result.get('filename', 'unknown')
            page = result.get('page', '')

            page_str = f" (p{page})" if page else ""

            if status == 'success':
                success_count += 1
                logger.info(f"[{i+1}/{total_files}] ✓ {filename}{page_str} -> {result.get('vehicle_no')}, {result.get('vin')}")

                # Append to CSV
                row = [
                    filename,
                    page if page else '',
                    result.get('vehicle_no', ''),
                    result.get('vehicle_type', ''),
                    result.get('model_name', ''),
                    result.get('vehicle_format', ''),
                    result.get('model_year', ''),
                    result.get('vin', ''),
                    "Valid" if result.get('vin_valid') else "Invalid",
                    result.get('engine_type', ''),
                    result.get('owner_name', ''),
                    result.get('owner_address', ''),
                    result.get('length_mm', ''),
                    result.get('width_mm', ''),
                    result.get('height_mm', ''),
                    result.get('total_weight_kg', ''),
                    result.get('displacement_cc', ''),
                    result.get('passenger_capacity', ''),
                    result.get('fuel_type', ''),
                    result.get('first_registration_date', ''),
                    result.get('usage_type', ''),
                    result.get('source_path', ''),
                    result.get('process_time', '')
                ]
                with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

            elif status == 'skip':
                skip_count += 1
                logger.info(f"[{i+1}/{total_files}] - {filename}{page_str}: SKIP ({result.get('reason')})")
            else:
                error_count += 1
                logger.warning(f"[{i+1}/{total_files}] ✗ {filename}{page_str}: {result.get('error')}")

    elapsed = time.time() - start_time
    avg_time = elapsed / total_files if total_files > 0 else 0

    logger.info(f"\n{'='*50}")
    logger.info(f"Processing complete!")
    logger.info(f"  Total items: {total_files}")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Skipped: {skip_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Total time: {elapsed:.1f}s")
    logger.info(f"  Avg time per item: {avg_time:.1f}s")
    logger.info(f"  CSV output: {csv_path}")
    logger.info(f"{'='*50}")

    return csv_path


def process_single_file_v2(args):
    """
    Process a single file (v2 - supports page number for multi-page PDFs).
    """
    import os
    import warnings
    import logging as log
    import tempfile

    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['GLOG_minloglevel'] = '3'
    log.getLogger('ppocr').setLevel(log.ERROR)

    # Unpack args
    if len(args) == 5:
        filepath, page_num, output_dir, dpi, use_structured = args
    elif len(args) == 4:
        filepath, output_dir, dpi, use_structured = args
        page_num = None
    else:
        filepath, output_dir, dpi = args
        page_num = None
        use_structured = True

    filename = os.path.basename(filepath)

    try:
        from paddleocr import PaddleOCR
        from PIL import Image

        # Initialize OCR
        ocr = PaddleOCR(
            lang='korean',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name='PP-OCRv5_mobile_det',
            text_recognition_model_name='korean_PP-OCRv5_mobile_rec',
            text_det_limit_side_len=1920,
            text_det_limit_type='max',
            text_det_box_thresh=0.5,
        )

        from src.parser.car_registration import CarRegistrationParser
        from src.parser.structured_parser import StructuredRegistrationParser
        from src.parser.template_extractor import TemplateExtractor
        from src.validator.vin_validator import VINValidator

        legacy_parser = CarRegistrationParser()
        structured_parser = StructuredRegistrationParser()
        template_extractor = TemplateExtractor()
        vin_validator = VINValidator()

        if not os.path.exists(filepath):
            return {'status': 'skip', 'reason': 'file_not_found', 'filename': filename}

        # Helper: sanitize filename for temp paths
        import re
        import uuid
        safe_basename = re.sub(r'[^\w\-.]', '_', os.path.splitext(filename)[0])
        unique_id = uuid.uuid4().hex[:8]
        temp_path = None

        # Handle PDF vs Image
        if filepath.lower().endswith('.pdf'):
            from pdf2image import convert_from_path
            page = page_num if page_num else 1
            pages = convert_from_path(filepath, dpi=dpi, first_page=page, last_page=page)
            if not pages:
                return {'status': 'skip', 'reason': 'pdf_empty', 'filename': filename, 'page': page}

            temp_path = os.path.join(tempfile.gettempdir(), f"{safe_basename}_{unique_id}_p{page}.jpg")
            page_img = pages[0]
            if page_img.mode in ('RGBA', 'LA', 'P'):
                page_img = page_img.convert('RGB')
            page_img.save(temp_path, 'JPEG', quality=85)
            pages[0].close()
            image_path = temp_path
        else:
            with Image.open(filepath) as img:
                width, height = img.size
                needs_conversion = img.mode in ('RGBA', 'LA', 'P')
                needs_resize = width > 2048 or height > 2048

                if needs_resize or needs_conversion:
                    if needs_conversion:
                        img = img.convert('RGB')
                    if needs_resize:
                        if width > height:
                            new_width = 2048
                            new_height = int(height * (2048 / width))
                        else:
                            new_height = 2048
                            new_width = int(width * (2048 / height))
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    temp_path = os.path.join(tempfile.gettempdir(), f"{safe_basename}_{unique_id}_proc.jpg")
                    img.save(temp_path, 'JPEG', quality=85)
                    image_path = temp_path
                else:
                    image_path = filepath

        # Run OCR
        result = ocr.predict(image_path)
        res = result[0].json['res']
        texts = res.get('rec_texts', [])
        scores = res.get('rec_scores', [])

        if not texts:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return {'status': 'skip', 'reason': 'no_text', 'filename': filename, 'page': page_num}

        full_text = '\n'.join(texts)

        # Verify document type
        if not structured_parser.verify_document_type(full_text):
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return {'status': 'skip', 'reason': 'not_registration', 'filename': filename, 'page': page_num}

        # Create raw result format
        raw_result = []
        boxes = res.get('rec_polys', [])
        for i, (text, score) in enumerate(zip(texts, scores)):
            box = boxes[i] if i < len(boxes) else [[0, 0], [0, 0], [0, 0], [0, 0]]
            raw_result.append([box, (text, score)])

        hybrid_result = {
            'google': {'text': full_text, 'avg_conf': sum(scores) / len(scores) if scores else 0.0, 'annotation': None},
            'paddle': {'result': raw_result, 'text_lines': texts}
        }

        # Parse
        parsed_data = legacy_parser.parse_hybrid(hybrid_result, filename=filename)
        confidences = parsed_data.get('confidences', {})

        if use_structured:
            pattern_data = structured_parser.extract_by_circled_patterns(full_text)
            pattern_data.pop('confidences', {})

            with Image.open(image_path) as img:
                img_width, img_height = img.size
            template_data = template_extractor.extract_from_boxes(raw_result, img_width, img_height)
            template_data.pop('confidences', {})

            supplement_fields = [
                'vehicle_no', 'vehicle_type', 'model_name', 'vehicle_format',
                'model_year', 'vin', 'engine_type', 'owner_name', 'owner_address',
                'spec_no', 'length_mm', 'width_mm', 'height_mm', 'total_weight_kg',
                'displacement_cc', 'rated_output', 'passenger_capacity', 'max_load_kg',
                'cylinders', 'fuel_type', 'type_approval_no',
                'first_registration_date', 'usage_type'
            ]

            for field in supplement_fields:
                if not parsed_data.get(field) and pattern_data.get(field):
                    parsed_data[field] = pattern_data[field]
            for field in supplement_fields:
                if not parsed_data.get(field) and template_data.get(field):
                    parsed_data[field] = template_data[field]

        # Validate VIN
        vin = parsed_data.get('vin')
        vehicle_no = parsed_data.get('vehicle_no')
        is_valid, _ = vin_validator.validate(vin)

        fuel_type = parsed_data.get('fuel_type', 'Unknown')

        # Cleanup temp
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            'status': 'success',
            'filename': filename,
            'page': page_num,
            'vehicle_no': vehicle_no,
            'vehicle_type': parsed_data.get('vehicle_type', ''),
            'model_name': parsed_data.get('model_name', ''),
            'vehicle_format': parsed_data.get('vehicle_format', ''),
            'model_year': parsed_data.get('model_year', ''),
            'vin': vin,
            'vin_valid': is_valid,
            'engine_type': parsed_data.get('engine_type', ''),
            'owner_name': parsed_data.get('owner_name', ''),
            'owner_address': parsed_data.get('owner_address', ''),
            'length_mm': parsed_data.get('length_mm', ''),
            'width_mm': parsed_data.get('width_mm', ''),
            'height_mm': parsed_data.get('height_mm', ''),
            'total_weight_kg': parsed_data.get('total_weight_kg', ''),
            'displacement_cc': parsed_data.get('displacement_cc', ''),
            'passenger_capacity': parsed_data.get('passenger_capacity', ''),
            'fuel_type': fuel_type,
            'first_registration_date': parsed_data.get('first_registration_date', ''),
            'usage_type': parsed_data.get('usage_type', ''),
            'source_path': filepath,
            'process_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    except Exception as e:
        import traceback
        return {'status': 'error', 'filename': filename, 'page': page_num, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='Fast parallel processing of vehicle registration files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan directory (with archive extraction)
  python process_local_fast.py --input-dir /path/to/folder

  # Process from file list
  python process_local_fast.py --file-list files.txt

  # Limit to first 10 files
  python process_local_fast.py --input-dir /path/to/folder --max 10

  # Skip archive extraction
  python process_local_fast.py --input-dir /path/to/folder --no-extract
        """
    )
    parser.add_argument('--input-dir', '-i',
                        help='Input directory to scan (supports nested archives)')
    parser.add_argument('--file-list', '-f',
                        help='Path to file list (one file per line)')
    parser.add_argument('--output-dir', '-o',
                        default='/Users/ssh/Documents/Develope/OCR_Multi_02/output',
                        help='Output directory for results')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting index')
    parser.add_argument('--max', type=int, default=None,
                        help='Maximum files to process')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: auto)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for PDF conversion (default: 150)')
    parser.add_argument('--legacy', action='store_true',
                        help='Use legacy parser instead of structured parser')
    parser.add_argument('--no-extract', action='store_true',
                        help='Do not extract archives')

    args = parser.parse_args()

    # Collect files to process
    files_to_process = []
    temp_dir = None

    if args.input_dir:
        # Scan directory
        if not os.path.isdir(args.input_dir):
            logger.error(f"Directory not found: {args.input_dir}")
            sys.exit(1)

        temp_dir = tempfile.mkdtemp(prefix='ocr_extract_')
        files_to_process = scan_directory(
            args.input_dir,
            extract_archives=not args.no_extract,
            temp_dir=temp_dir
        )
        logger.info(f"Found {len(files_to_process)} files to process")

    elif args.file_list:
        # Read from file list
        if not os.path.isfile(args.file_list):
            logger.error(f"File list not found: {args.file_list}")
            sys.exit(1)

        with open(args.file_list, 'r', encoding='utf-8') as f:
            files_to_process = [normalize_path(line.strip()) for line in f if line.strip()]
        logger.info(f"Loaded {len(files_to_process)} files from list")

    else:
        # Default: use dropbox file list if exists
        default_list = '/Users/ssh/Documents/Develope/OCR_Multi_02/dropbox_registration_files.txt'
        if os.path.isfile(default_list):
            with open(default_list, 'r', encoding='utf-8') as f:
                files_to_process = [normalize_path(line.strip()) for line in f if line.strip()]
            logger.info(f"Using default file list: {len(files_to_process)} files")
        else:
            logger.error("Please specify --input-dir or --file-list")
            sys.exit(1)

    if not files_to_process:
        logger.error("No files to process!")
        sys.exit(1)

    # Process files
    csv_path = process_files_from_list(
        files=files_to_process,
        output_dir=args.output_dir,
        start_idx=args.start,
        max_files=args.max,
        num_workers=args.workers,
        dpi=args.dpi,
        use_structured=not args.legacy
    )

    # Cleanup temp directory
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean temp directory: {e}")


if __name__ == "__main__":
    main()
