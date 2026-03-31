# -*- coding: utf-8 -*-
"""
Gradio web app for Korean Vehicle Registration Certificate OCR.
Deployed on HuggingFace Spaces.
"""
import os
os.environ["GRADIO_SSR_MODE"] = "false"
import logging
import traceback
import gradio as gr
import pandas as pd
from datetime import datetime

from src.processor import process_batch, results_to_rows, warmup
from src.storage.excel_writer import ExcelWriter
from src.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Excel column headers
HEADERS = [
    '차량번호', '업체명', '차대번호', '차명', '연식',
    '차량등록일', '차종', '길이(mm)', '너비(mm)', '높이(mm)',
    '총중량(kg)', '승차정원', '연료', '출고가격(원)',
]

SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.pdf')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def get_file_path(f):
    """Extract file path from various Gradio file object formats."""
    if isinstance(f, str):
        return f
    elif hasattr(f, 'name'):
        return f.name
    elif hasattr(f, 'path'):
        return f.path
    return str(f)


def build_file_gallery(files):
    """Build a list of (filepath, label) for the file selector and preview the first image."""
    if not files:
        return gr.Dropdown(choices=[], value=None, visible=False), None

    choices = []
    first_image = None
    for f in files:
        fp = get_file_path(f)
        fname = os.path.basename(fp)
        if fname.lower().endswith(SUPPORTED_EXTENSIONS):
            choices.append((fname, fp))
            if first_image is None and fname.lower().endswith(IMAGE_EXTENSIONS):
                first_image = fp

    if not choices:
        return gr.Dropdown(choices=[], value=None, visible=False), None

    # Auto-preview: first image, or first PDF converted to image
    first_preview = first_image or preview_selected_file(choices[0][1])

    return (
        gr.Dropdown(choices=choices, value=choices[0][1], visible=True),
        first_preview,
    )


def preview_selected_file(file_path):
    """Show preview for the selected file."""
    if not file_path or not os.path.exists(file_path):
        return None

    if file_path.lower().endswith(IMAGE_EXTENSIONS):
        return file_path

    # For PDF, convert first page to image for preview
    if file_path.lower().endswith('.pdf'):
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(file_path, first_page=1, last_page=1, dpi=150)
            if images:
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                images[0].save(tmp.name, 'PNG')
                return tmp.name
        except Exception as e:
            logger.warning(f"PDF preview failed: {e}")

    return None


def run_ocr(files, progress=gr.Progress()):
    """
    Main processing function called by Gradio.
    """
    try:
        if not files:
            return pd.DataFrame(), None, "파일을 업로드해주세요."

        # Build file list
        file_list = []
        logger.info(f"Received {len(files)} files, type: {type(files[0]) if files else 'none'}")

        for f in files:
            file_path = get_file_path(f)
            filename = os.path.basename(file_path)
            logger.info(f"File: {filename}, path: {file_path}, exists: {os.path.exists(file_path)}")

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

        # Add details to summary
        for r in results:
            if r['status'] in ('error', 'skipped'):
                summary += f"\n  - {r['filename']}: {r.get('message', 'unknown')}"
            elif r['status'] == 'success':
                d = r['data']
                vin_info = d.get('vin', '?')
                vin_msg = d.get('vin_message', '')
                if vin_msg and vin_info != '?':
                    vin_info += f" [{vin_msg}]"
                summary += f"\n  - {r['filename']}: 차량번호={d.get('vehicle_no','?')} 차대번호={vin_info}"
                ocr_preview = d.get('_ocr_preview', '')
                if ocr_preview:
                    summary += f"\n    [OCR 원문 미리보기] {ocr_preview[:200]}"

        # Convert to rows
        rows = results_to_rows(results)

        if not rows:
            return pd.DataFrame(columns=HEADERS), None, summary + "\n\n인식된 자동차등록증이 없습니다."

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

    except Exception as e:
        error_msg = f"시스템 오류: {str(e)}\n\n{traceback.format_exc()}"
        logger.error(error_msg)
        return pd.DataFrame(), None, error_msg


# Build Gradio UI
with gr.Blocks(
    title="자동차등록증 OCR",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("# 자동차등록증 OCR 시스템 <sub style='color:gray;font-weight:normal'>v36</sub>")
    gr.Markdown("자동차등록증 이미지 또는 PDF를 업로드하면 OCR로 정보를 추출하여 엑셀 파일로 저장합니다.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="파일 업로드 (JPG, PNG, PDF)",
                file_count="multiple",
                file_types=[".jpg", ".jpeg", ".png", ".pdf"],
            )
            file_selector = gr.Dropdown(
                label="파일 선택 (미리보기)",
                choices=[],
                visible=False,
                interactive=True,
            )
        with gr.Column(scale=1):
            image_preview = gr.Image(
                label="이미지 미리보기",
                type="filepath",
                interactive=False,
                height=400,
            )

    run_btn = gr.Button("OCR 처리 시작", variant="primary", size="lg")

    summary_text = gr.Textbox(label="처리 결과", interactive=False, lines=5)

    result_table = gr.Dataframe(
        label="결과 미리보기",
        headers=HEADERS,
        interactive=False,
    )

    excel_download = gr.File(label="엑셀 다운로드")

    # When files are uploaded, populate the file selector and show first image
    file_input.change(
        fn=build_file_gallery,
        inputs=[file_input],
        outputs=[file_selector, image_preview],
    )

    # When a file is selected from dropdown, show its preview
    file_selector.change(
        fn=preview_selected_file,
        inputs=[file_selector],
        outputs=[image_preview],
    )

    run_btn.click(
        fn=run_ocr,
        inputs=[file_input],
        outputs=[result_table, excel_download, summary_text],
    )

# Pre-initialize Korean OCR engine (layout engine loads lazily on first use)
warmup()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
