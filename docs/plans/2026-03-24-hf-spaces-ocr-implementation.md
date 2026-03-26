# HuggingFace Spaces OCR Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deploy a Gradio web app on HuggingFace Spaces that processes Korean vehicle registration certificate images/PDFs via PaddleOCR and returns results as a downloadable Excel file.

**Architecture:** Single Gradio app (`app.py`) calls `src/processor.py` which orchestrates PaddleOCR → parse → validate → Excel pipeline. Google Vision is removed entirely. PaddleOCR runs locally on the HF Spaces container.

**Tech Stack:** Python 3.10+, Gradio 4.x, PaddleOCR 2.x, openpyxl, Pillow, pdf2image

---

### Task 1: Clean up repository — remove __pycache__ from git and unused files

**Files:**
- Delete: all `src/**/__pycache__/` directories from git tracking
- Delete: `src/ocr/engine.py` (Google Vision)
- Delete: `src/ocr/hybrid_engine.py` (hybrid logic)
- Delete: `src/ocr/local_engine.py` (duplicate of paddle_engine)
- Delete: `src/storage/gsheets.py` (Google Sheets)
- Delete: `src/storage/vin_logger.py` (CSV logger)
- Delete: `src/tools/` entire directory (debug scripts)
- Delete: `src/parser/template_extractor.py` (unused)
- Delete: `src/parser/structured_parser.py` (unused)
- Modify: `.gitignore`

**Step 1: Remove __pycache__ from git tracking**

```bash
git rm -r --cached src/__pycache__/ src/ocr/__pycache__/ src/parser/__pycache__/ src/storage/__pycache__/ src/validator/__pycache__/ src/tools/__pycache__/
```

**Step 2: Delete unused source files**

```bash
git rm src/ocr/engine.py
git rm src/ocr/hybrid_engine.py
git rm src/ocr/local_engine.py
git rm src/storage/gsheets.py
git rm src/storage/vin_logger.py
git rm -r src/tools/
git rm src/parser/template_extractor.py
git rm src/parser/structured_parser.py
```

**Step 3: Verify .gitignore has __pycache__**

`.gitignore` already contains `__pycache__/` — verify no changes needed.

**Step 4: Remove processed data file from tracking**

```bash
git rm --cached processed/vin_recognition_stats.csv
```

Add to `.gitignore`:
```
processed/
```

(Already there — verify.)

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove unused files and __pycache__ from tracking

Remove Google Vision engine, hybrid engine, Google Sheets storage,
VIN logger, debug tools, and cached bytecode files.
Prepare for HuggingFace Spaces deployment with PaddleOCR only."
```

---

### Task 2: Simplify config.py — remove Google API dependencies

**Files:**
- Modify: `src/config.py` (full rewrite)

**Step 1: Write the new config.py**

Replace entire `src/config.py` with:

```python
# -*- coding: utf-8 -*-
import os


class Config:
    # Base directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

    # OCR Settings
    OCR_LANGUAGE = 'korean'
    PDF_DPI = int(os.getenv('PDF_DPI', '150'))
    MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '2048'))

    # Processing Settings
    GC_INTERVAL = int(os.getenv('GC_INTERVAL', '10'))

    @classmethod
    def ensure_dirs(cls):
        """Create output directory if it doesn't exist."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
```

This removes: `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_SHEET_ID`, `INPUT_DIR`, `PROCESSED_DIR`, `FAILED_DIR`, `ENABLE_PADDLE`, `BATCH_SIZE`, `RATE_LIMIT_SECONDS`, `dotenv` dependency, `validate()` method.

**Step 2: Run a quick import check**

```bash
cd /Users/ssh/Documents/Develope/OCR_Multi_02
python -c "from src.config import Config; print(Config.BASE_DIR); Config.ensure_dirs(); print('OK')"
```

Expected: prints the base directory path and "OK".

**Step 3: Commit**

```bash
git add src/config.py
git commit -m "refactor: simplify config for HF Spaces deployment

Remove Google API settings, dotenv dependency, file-move directories.
Keep only OCR and output settings needed for PaddleOCR-only mode."
```

---

### Task 3: Add parse_single() method to CarRegistrationParser

The existing `parse()` method works on raw text. We need a convenience method that takes OCR text + filename and returns a complete result dict with fuel type and filename fallbacks — replacing what `parse_hybrid()` did but without Google Vision.

**Files:**
- Modify: `src/parser/car_registration.py:139` (add method after `parse()`)
- Test: `tests/test_parser.py`

**Step 1: Write the failing test**

Add to `tests/test_parser.py`:

```python
def test_parse_single_returns_fuel_type(self):
    """parse_single should determine fuel type from filename."""
    result = self.parser.parse_single(self.decoded_text, filename="강원70자1016_전기.pdf")
    self.assertEqual(result['fuel_type'], 'Electric')

def test_parse_single_returns_all_fields(self):
    """parse_single should return all 13 output fields."""
    required = [
        'vehicle_no', 'owner_name', 'vin', 'model_name', 'model_year',
        'registration_date', 'vehicle_type', 'length_mm', 'width_mm',
        'height_mm', 'total_weight_kg', 'passenger_capacity', 'fuel_type',
    ]
    result = self.parser.parse_single(self.decoded_text, filename="test.pdf")
    for field in required:
        self.assertIn(field, result, f"Missing field: {field}")

def test_parse_single_vehicle_no_from_filename(self):
    """parse_single should extract vehicle_no from filename as fallback."""
    # Use text without vehicle_no label
    text = "차대번호 : KL1T1234567890123"
    result = self.parser.parse_single(text, filename="경북70자6310.pdf")
    self.assertEqual(result['vehicle_no'], '경북70자6310')
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/ssh/Documents/Develope/OCR_Multi_02
python -m pytest tests/test_parser.py -v -k "parse_single"
```

Expected: FAIL with `AttributeError: 'CarRegistrationParser' object has no attribute 'parse_single'`

**Step 3: Implement parse_single()**

Add this method to `CarRegistrationParser` in `src/parser/car_registration.py` after the `parse()` method (after line 184):

```python
def parse_single(self, ocr_text, filename=None):
    """
    Parse OCR text from a single engine and return complete result.
    Replaces parse_hybrid() for PaddleOCR-only mode.

    Args:
        ocr_text: Full OCR text string
        filename: Original filename for fallback extraction

    Returns:
        dict with all 13 output fields
    """
    result = self.parse(ocr_text) if ocr_text else {}

    # Fallback: extract vehicle_no from filename if OCR failed
    ocr_vehicle_no = result.get('vehicle_no')
    if filename and (not ocr_vehicle_no or not self._is_valid_vehicle_no(ocr_vehicle_no)):
        filename_vehicle_no = self._extract_vehicle_no_from_filename(filename)
        if filename_vehicle_no:
            logging.info(f"Vehicle No from filename: {filename_vehicle_no} (OCR was: '{ocr_vehicle_no}')")
            result['vehicle_no'] = filename_vehicle_no

    # Determine fuel type
    fuel_type = self._determine_fuel_type(
        filename=filename,
        ocr_text=ocr_text,
        model_name=result.get('model_name'),
        engine_type=result.get('engine_type'),
    )
    result['fuel_type'] = fuel_type

    return result
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_parser.py -v
```

Expected: ALL PASS (existing tests + 3 new tests)

**Step 5: Commit**

```bash
git add src/parser/car_registration.py tests/test_parser.py
git commit -m "feat: add parse_single() for PaddleOCR-only mode

Replaces parse_hybrid() without Google Vision dependency.
Includes filename fallback for vehicle_no and fuel type extraction."
```

---

### Task 4: Create processor.py — core processing pipeline

**Files:**
- Create: `src/processor.py`
- Test: `tests/test_processor.py`

**Step 1: Write the failing test**

Create `tests/test_processor.py`:

```python
# -*- coding: utf-8 -*-
import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
from src.processor import process_single_file, process_batch


class TestProcessSingleFile(unittest.TestCase):
    """Test process_single_file with mocked OCR engine."""

    @patch('src.processor.get_ocr_engine')
    @patch('src.processor.ImagePreprocessor')
    def test_returns_dict_with_all_fields(self, mock_preprocessor_cls, mock_get_engine):
        """process_single_file should return a dict with status and data."""
        # Mock preprocessor
        mock_prep = MagicMock()
        mock_prep.load_image.return_value = ['/tmp/fake_image.jpg']
        mock_preprocessor_cls.return_value = mock_prep

        # Mock OCR engine
        mock_engine = MagicMock()
        mock_engine.detect_text.return_value = {
            'text': '자동차등록증\n차대번호 : KL1T1234567890123',
            'lines': [('자동차등록증', 0.95), ('차대번호 : KL1T1234567890123', 0.92)],
            'avg_confidence': 0.93,
        }
        mock_get_engine.return_value = mock_engine

        result = process_single_file('/tmp/fake.jpg', 'fake.jpg')

        self.assertEqual(result['status'], 'success')
        self.assertIn('data', result)
        self.assertIn('vin', result['data'])

    @patch('src.processor.get_ocr_engine')
    @patch('src.processor.ImagePreprocessor')
    def test_returns_error_on_non_registration(self, mock_preprocessor_cls, mock_get_engine):
        """process_single_file should return skip status for non-registration docs."""
        mock_prep = MagicMock()
        mock_prep.load_image.return_value = ['/tmp/fake_image.jpg']
        mock_preprocessor_cls.return_value = mock_prep

        mock_engine = MagicMock()
        mock_engine.detect_text.return_value = {
            'text': 'This is a random document with no registration keywords',
            'lines': [('This is a random document', 0.9)],
            'avg_confidence': 0.9,
        }
        mock_get_engine.return_value = mock_engine

        result = process_single_file('/tmp/fake.jpg', 'fake.jpg')

        self.assertEqual(result['status'], 'skipped')


class TestProcessBatch(unittest.TestCase):
    """Test process_batch orchestration."""

    @patch('src.processor.process_single_file')
    def test_batch_returns_list(self, mock_process):
        """process_batch should return a list of results."""
        mock_process.return_value = {'status': 'success', 'data': {}, 'filename': 'a.jpg'}
        files = [('/tmp/a.jpg', 'a.jpg'), ('/tmp/b.jpg', 'b.jpg')]
        results = process_batch(files)
        self.assertEqual(len(results), 2)


if __name__ == '__main__':
    unittest.main()
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_processor.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.processor'`

**Step 3: Implement processor.py**

Create `src/processor.py`:

```python
# -*- coding: utf-8 -*-
"""
Core processing pipeline for vehicle registration OCR.
Orchestrates: image preprocessing -> PaddleOCR -> parsing -> validation -> results.
"""
import os
import gc
import logging
from src.ocr.paddle_engine import LocalPaddleEngine
from src.ocr.preprocessor import ImagePreprocessor
from src.parser.car_registration import CarRegistrationParser
from src.validator.vin_validator import VINValidator
from src.config import Config

logger = logging.getLogger(__name__)

# Singleton instances
_ocr_engine = None
_parser = CarRegistrationParser()
_validator = VINValidator()
_preprocessor = ImagePreprocessor(dpi=Config.PDF_DPI, max_size=Config.MAX_IMAGE_SIZE)


def get_ocr_engine():
    """Get or create the PaddleOCR engine singleton."""
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = LocalPaddleEngine(lang=Config.OCR_LANGUAGE, enable_paddle=True)
    return _ocr_engine


def process_single_file(file_path, filename):
    """
    Process a single image/PDF file through the OCR pipeline.

    Args:
        file_path: Absolute path to the file
        filename: Original filename (used for fallback extraction)

    Returns:
        dict with keys:
            - status: 'success' | 'skipped' | 'error'
            - filename: original filename
            - data: parsed fields dict (if success)
            - message: error/skip reason (if not success)
    """
    try:
        engine = get_ocr_engine()

        # 1. Preprocess image/PDF
        image_paths = _preprocessor.load_image(file_path)
        if not image_paths:
            return {'status': 'error', 'filename': filename, 'data': {}, 'message': 'Failed to load image'}

        # Process first page only (vehicle registration is single page)
        image_path = image_paths[0]

        try:
            # 2. Run PaddleOCR
            ocr_result = engine.detect_text(image_path)
            ocr_text = ocr_result.get('text', '')

            if not ocr_text:
                return {'status': 'error', 'filename': filename, 'data': {}, 'message': 'No text detected'}

            # 3. Verify document type
            if not _parser.verify_document_type(ocr_text):
                return {'status': 'skipped', 'filename': filename, 'data': {}, 'message': 'Not a vehicle registration certificate'}

            # 4. Parse
            parsed_data = _parser.parse_single(ocr_text, filename=filename)

            # 5. Validate VIN
            vin = parsed_data.get('vin')
            is_valid, validation_msg = _validator.validate(vin)
            parsed_data['vin_valid'] = is_valid
            parsed_data['vin_message'] = validation_msg

            logger.info(f"Successfully processed: {filename}")
            return {'status': 'success', 'filename': filename, 'data': parsed_data, 'message': 'OK'}

        finally:
            # Cleanup temp files from PDF conversion
            if filename.lower().endswith('.pdf'):
                ImagePreprocessor.cleanup_temp_files(image_paths, file_path)

    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return {'status': 'error', 'filename': filename, 'data': {}, 'message': str(e)}


def process_batch(file_list, progress_callback=None):
    """
    Process a batch of files.

    Args:
        file_list: List of (file_path, filename) tuples
        progress_callback: Optional callable(current, total) for progress updates

    Returns:
        List of result dicts from process_single_file
    """
    results = []
    total = len(file_list)

    for i, (file_path, filename) in enumerate(file_list):
        result = process_single_file(file_path, filename)
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total)

        # Memory management
        if (i + 1) % Config.GC_INTERVAL == 0:
            gc.collect()

    return results


def results_to_rows(results):
    """
    Convert processing results to rows for Excel output.

    Args:
        results: List of result dicts from process_batch

    Returns:
        List of lists, each inner list is one Excel row (13 columns)
    """
    rows = []
    for r in results:
        if r['status'] != 'success':
            continue
        d = r['data']
        rows.append([
            d.get('vehicle_no', ''),
            d.get('owner_name', ''),
            d.get('vin', ''),
            d.get('model_name', ''),
            d.get('model_year', ''),
            d.get('registration_date', ''),
            d.get('vehicle_type', ''),
            d.get('length_mm', ''),
            d.get('width_mm', ''),
            d.get('height_mm', ''),
            d.get('total_weight_kg', ''),
            d.get('passenger_capacity', ''),
            d.get('fuel_type', ''),
        ])
    return rows
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_processor.py -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/processor.py tests/test_processor.py
git commit -m "feat: add processor.py — core OCR processing pipeline

Orchestrates PaddleOCR -> parse -> validate flow.
Provides process_single_file, process_batch, and results_to_rows."
```

---

### Task 5: Create app.py — Gradio web interface

**Files:**
- Create: `app.py` (project root)

**Step 1: Create app.py**

```python
# -*- coding: utf-8 -*-
"""
Gradio web app for Korean Vehicle Registration Certificate OCR.
Deployed on HuggingFace Spaces.
"""
import os
import tempfile
import logging
import gradio as gr
import pandas as pd
from datetime import datetime

from src.processor import process_batch, results_to_rows
from src.storage.excel_writer import ExcelWriter
from src.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Excel column headers
HEADERS = [
    '차량번호', '업체명', '차대번호', '차명', '연식',
    '차량등록일', '차종', '길이(mm)', '너비(mm)', '높이(mm)',
    '총중량(kg)', '승차정원', '연료',
]

SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.pdf')


def run_ocr(files, progress=gr.Progress()):
    """
    Main processing function called by Gradio.

    Args:
        files: List of uploaded file paths from gr.File
        progress: Gradio progress tracker

    Returns:
        Tuple of (DataFrame for preview, Excel file path for download, summary text)
    """
    if not files:
        return pd.DataFrame(), None, "파일을 업로드해주세요."

    # Build file list
    file_list = []
    for f in files:
        file_path = f if isinstance(f, str) else f.name
        filename = os.path.basename(file_path)
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            file_list.append((file_path, filename))

    if not file_list:
        return pd.DataFrame(), None, "지원되는 파일이 없습니다. (JPG, PNG, PDF)"

    total = len(file_list)
    logger.info(f"Processing {total} files...")

    # Process with progress callback
    def progress_cb(current, total_count):
        progress(current / total_count, desc=f"{current}/{total_count} 처리 중...")

    results = process_batch(file_list, progress_callback=progress_cb)

    # Build summary
    success = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')
    summary = f"처리 완료: {total}개 파일 중 {success}개 성공"
    if skipped > 0:
        summary += f", {skipped}개 건너뜀 (자동차등록증 아님)"
    if errors > 0:
        summary += f", {errors}개 실패"

    # Convert to rows
    rows = results_to_rows(results)

    if not rows:
        return pd.DataFrame(columns=HEADERS), None, summary + "\n인식된 자동차등록증이 없습니다."

    # Create DataFrame for preview
    df = pd.DataFrame(rows, columns=HEADERS)

    # Write Excel file
    Config.ensure_dirs()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    excel_path = os.path.join(Config.OUTPUT_DIR, f'OCR_결과_{timestamp}.xlsx')
    writer = ExcelWriter(excel_path)
    for row in rows:
        writer.append_row(row)
    writer.close()

    logger.info(f"Results saved to: {excel_path}")

    return df, excel_path, summary


# Build Gradio UI
with gr.Blocks(
    title="자동차등록증 OCR",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("# 자동차등록증 OCR 시스템")
    gr.Markdown("자동차등록증 이미지 또는 PDF를 업로드하면 OCR로 정보를 추출하여 엑셀 파일로 저장합니다.")

    with gr.Row():
        file_input = gr.File(
            label="파일 업로드 (JPG, PNG, PDF)",
            file_count="multiple",
            file_types=[".jpg", ".jpeg", ".png", ".pdf"],
        )

    run_btn = gr.Button("OCR 처리 시작", variant="primary", size="lg")

    summary_text = gr.Textbox(label="처리 결과", interactive=False)

    result_table = gr.Dataframe(
        label="결과 미리보기",
        headers=HEADERS,
        interactive=False,
    )

    excel_download = gr.File(label="엑셀 다운로드")

    run_btn.click(
        fn=run_ocr,
        inputs=[file_input],
        outputs=[result_table, excel_download, summary_text],
    )

if __name__ == "__main__":
    demo.launch()
```

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('app.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Gradio web app for HuggingFace Spaces

File upload -> PaddleOCR -> result table preview -> Excel download.
Supports multiple JPG/PNG/PDF files."
```

---

### Task 6: Update requirements.txt and create packages.txt

**Files:**
- Modify: `requirements.txt`
- Create: `packages.txt`

**Step 1: Write requirements.txt**

```
paddlepaddle==3.0.0
paddleocr==3.0.0
gradio>=4.0
openpyxl==3.1.2
Pillow>=10.0
pdf2image>=1.16
pandas>=2.0
```

**Step 2: Write packages.txt**

```
poppler-utils
libgl1-mesa-glx
```

**Step 3: Commit**

```bash
git add requirements.txt packages.txt
git commit -m "chore: update dependencies for HF Spaces deployment

Add Gradio, PaddleOCR, pdf2image. Add system packages for poppler and OpenCV."
```

---

### Task 7: Update README.md with HF Spaces metadata

**Files:**
- Modify: `README.md`

**Step 1: Write README.md**

```markdown
---
title: 자동차등록증 OCR
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
---

# 자동차등록증 OCR 시스템

한국 자동차등록증(Vehicle Registration Certificate) 이미지/PDF를 OCR 처리하여 주요 정보를 엑셀로 추출합니다.

## 사용 방법

1. 자동차등록증 이미지(JPG, PNG) 또는 PDF 파일을 업로드합니다.
2. "OCR 처리 시작" 버튼을 클릭합니다.
3. 처리가 완료되면 결과를 미리보기하고 엑셀 파일을 다운로드합니다.

## 추출 항목

차량번호, 업체명, 차대번호, 차명, 연식, 차량등록일, 차종, 길이, 너비, 높이, 총중량, 승차정원, 연료

## 기술 스택

- **OCR Engine:** PaddleOCR (Korean)
- **UI:** Gradio
- **Platform:** HuggingFace Spaces
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add HF Spaces metadata and usage instructions"
```

---

### Task 8: Connect GitHub repo to HuggingFace Spaces and verify deployment

**Step 1: Add HF Spaces as a git remote**

```bash
git remote add hf https://huggingface.co/spaces/Esketch/OCR_Vehicle_01
```

**Step 2: Push to HuggingFace Spaces**

```bash
git push hf main
```

**Step 3: Verify deployment**

Open `https://huggingface.co/spaces/Esketch/OCR_Vehicle_01` in a browser.
Expected: Gradio UI loads with file upload area and "OCR 처리 시작" button.

**Step 4: Push to GitHub too**

```bash
git push origin main
```

**Step 5: Test with a sample file**

Upload a test vehicle registration certificate image and verify:
- OCR processes without error
- Result table shows extracted data
- Excel file downloads successfully

---

## Execution Notes

- Tasks 1-7 are code changes that can be done locally.
- Task 8 is deployment — requires HuggingFace credentials.
- If PaddleOCR version issues occur on HF Spaces, try `paddleocr>=2.7,<3.0` and `paddlepaddle>=2.5,<3.0` in requirements.txt.
- HF Spaces free tier has 16GB RAM and 2 vCPU — sufficient for PaddleOCR.
