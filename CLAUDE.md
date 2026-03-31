# OCR_Multi_02 - 한국 차량등록증 OCR 시스템

## 프로젝트 개요
한국 차량등록증 이미지를 OCR로 읽어 구조화된 데이터(차량번호, 업체명, 차대번호, 차명 등 13개 필드)를 추출하는 시스템. Gradio 웹 UI + HuggingFace Spaces 배포.

## 기술 스택
- **OCR 엔진**: PaddleOCR (paddlepaddle 2.6.2 + paddleocr 2.9.1)
- **Web UI**: Gradio
- **배포**: HuggingFace Spaces
- **언어**: Python
- **출력**: Excel (openpyxl), CSV (pandas)

## 아키텍처
```
app.py                  # Gradio 웹 인터페이스
src/
  config.py             # 설정 관리
  processor.py          # 배치 처리 오케스트레이션
  ocr/
    paddle_engine.py    # PaddleOCR 엔진 래퍼
    preprocessor.py     # 이미지 전처리 (회전, 노이즈 제거 등)
  parser/
    car_registration.py # OCR 텍스트 → 구조화 데이터 파싱
  storage/
    excel_writer.py     # Excel 출력
  validator/
    standards.py        # 차종/연료 등 표준값 매칭
    vin_validator.py    # 차대번호 검증
tests/
  test_parser.py        # 파서 테스트
  test_processor.py     # 프로세서 테스트
```

## 개발 관리 체계: 3-역할 분리

이 프로젝트는 세 가지 전문화된 역할로 개발을 관리한다.

### 역할 호출 방법
각 역할은 별도 Claude Code 세션에서 해당 역할의 CLAUDE.md를 참조하여 동작한다:
- **Orchestrator**: `CLAUDE.md` + `docs/roles/orchestrator.md` 참조
- **Full-stack Dev**: `CLAUDE.md` + `docs/roles/fullstack-dev.md` 참조
- **QAQC**: `CLAUDE.md` + `docs/roles/qaqc.md` 참조

### 역할 간 커뮤니케이션
- 모든 역할은 `docs/handoff/` 디렉토리를 통해 작업을 주고받는다
- Orchestrator가 작업 지시서를 작성 → Dev가 구현 → QAQC가 검증
- 각 핸드오프 문서는 날짜-제목 형식: `YYYY-MM-DD-<slug>.md`

## 코딩 컨벤션
- Python 3.9+ 호환
- 한글 주석/로깅 허용, 코드는 영문
- 로깅: `logging` 모듈 사용 (print 금지)
- 에러 처리: 개별 파일 실패가 배치 전체를 중단하지 않도록
- 테스트: `tests/` 디렉토리, pytest 사용

## 현재 로드맵
### Phase 1 (진행중): 폼 양식 학습 기반 인식 모델 - PP-Structure
- PP-StructureV2 레이아웃 분석 엔진 통합
- 테이블/키-밸류 구조 추출 → 기존 정규식 파서와 앙상블
- 지시서: `docs/handoff/2026-03-31-task-form-aware-model.md`

### Phase 2 (예정): 템플릿 매칭 + 앵커 기반 ROI 추출
### Phase 3 (장기): Document AI (LayoutLM) 검토

## 브랜치 전략
- `main`: 배포 브랜치 (HF Spaces 자동 배포)
- 기능 개발: `feat/<name>`, 버그 수정: `fix/<name>`
- QAQC 통과 후 main 머지
