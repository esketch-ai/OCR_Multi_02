# 자동차등록증 OCR 시스템 - HuggingFace Spaces 배포 디자인

## 목표

비개발자 실무 담당자가 매월 웹 브라우저에서 자동차등록증 이미지/PDF를 업로드하면 OCR 처리 후 결과 엑셀을 다운로드할 수 있는 시스템.

## 핵심 결정사항

- **플랫폼:** HuggingFace Spaces (`Esketch/OCR_Vehicle_01`)
- **UI:** Gradio (파일 업로드 → OCR → 엑셀 다운로드)
- **OCR 엔진:** PaddleOCR만 사용 (Google Vision 제거, API 비용/키 관리 없음)
- **원본 파일:** 건드리지 않음
- **출력:** 13개 컬럼 엑셀 (차량번호, 업체명, 차대번호, 차명, 연식, 차량등록일, 차종, 길이, 너비, 높이, 총중량, 승차정원, 연료)
- **배포:** GitHub repo를 HF Spaces에 직접 연결 (push = 자동 배포)

## 사용자 흐름

1. 담당자가 HF Spaces URL 접속
2. 자동차등록증 파일 드래그 앤 드롭 (복수 파일, JPG/PNG/PDF)
3. "OCR 처리 시작" 버튼 클릭
4. 진행 상태 표시 + 결과 테이블 미리보기
5. "엑셀 다운로드" 버튼으로 결과 파일 다운로드

## UI 구성

```
┌─────────────────────────────────────────────┐
│  자동차등록증 OCR 시스템                       │
│                                             │
│  [파일 업로드 영역 - 드래그 앤 드롭]            │
│  JPG, PNG, PDF 지원 / 최대 20개              │
│                                             │
│  [ OCR 처리 시작 ]                           │
│                                             │
│  처리 상태: 5/20 완료 (25%)                   │
│                                             │
│  [결과 미리보기 테이블]                        │
│                                             │
│  [ 엑셀 다운로드 ]                            │
│                                             │
│  처리 요약: 20개 중 18개 성공, 2개 실패         │
└─────────────────────────────────────────────┘
```

## 프로젝트 구조

```
app.py                  # Gradio 앱 진입점
src/
├── processor.py        # 핵심 처리 파이프라인 (신규)
├── config.py           # 설정 (간소화)
├── ocr/
│   ├── paddle_engine.py  # PaddleOCR 엔진
│   └── preprocessor.py   # 이미지 전처리
├── parser/
│   └── car_registration.py  # 파싱
├── validator/
│   └── vin_validator.py     # VIN 검증
└── storage/
    └── excel_writer.py      # 엑셀 저장
requirements.txt        # PaddleOCR + Gradio 의존성
packages.txt            # 시스템 패키지 (poppler-utils, libgl1-mesa-glx)
README.md               # HF Spaces 메타데이터
```

## 처리 파이프라인 (파일 1개당)

```
파일 업로드 → 이미지 전처리 → PaddleOCR → 문서 유형 확인 → 파싱 → VIN 검증 → 결과 반환
```

- `processor.py`의 `process_single_file(file_path, filename) -> dict`
- `processor.py`의 `process_batch(file_list, progress_callback) -> list[dict]`
- Google Vision 의존성 완전 제거
- rate limiting 제거 (로컬 OCR이므로 불필요)

## 구현 작업 목록

### 신규 생성
- `app.py` — Gradio UI
- `src/processor.py` — main.py에서 처리 로직 추출
- `packages.txt` — 시스템 패키지

### 수정
- `src/config.py` — Google API 설정 제거, 간소화
- `src/parser/car_registration.py` — `parse()` 직접 사용, hybrid 의존 제거
- `requirements.txt` — Spaces용 의존성
- `README.md` — HF Spaces 메타데이터

### 유지
- `src/ocr/paddle_engine.py`
- `src/ocr/preprocessor.py`
- `src/validator/vin_validator.py`
- `src/storage/excel_writer.py`

### 제거
- `src/ocr/engine.py` (Google Vision)
- `src/ocr/hybrid_engine.py` (하이브리드 로직)
- `src/storage/gsheets.py` (Google Sheets)
- `src/storage/vin_logger.py` (CSV 로그)
- `src/tools/` 전체 (디버그 스크립트)
- `src/ocr/local_engine.py` (중복)
- `__pycache__/` (git에서 제거)

## 배포 설정

### requirements.txt
```
paddlepaddle==3.0.0
paddleocr==3.0.0
gradio>=4.0
openpyxl==3.1.2
Pillow>=10.0
pdf2image>=1.16
python-dotenv==1.0.0
```

### packages.txt
```
poppler-utils
libgl1-mesa-glx
```

### README.md 메타데이터
```yaml
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
```
