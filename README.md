---
title: 자동차등록증 OCR
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.23.0"
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
