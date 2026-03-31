# Review: Phase 1 - PP-Structure 레이아웃 분석 엔진 통합

- **날짜**: 2026-03-31
- **대상**: main 브랜치 (미커밋 변경사항)
- **판정**: CONDITIONAL_PASS

## 검증 결과

| 항목 | 결과 | 비고 |
|------|------|------|
| 정적 분석 | OK | Python 문법 통과, print 없음, 시크릿 없음 |
| 테스트 | OK | 신규 14개 전부 통과, 기존 회귀 테스트 통과 |
| OCR 정확도 | N/A | PP-Structure 의존성 미설치로 실제 테스트 불가 |
| UI/UX | N/A | app.py 변경 없음 |
| 배포 호환성 | OK (조건부) | graceful fallback 동작 확인, HF Spaces 실배포 필요 |

## 상세 피드백

### Issue 1: `_match_label` 서브스트링 매칭 오탐 위험 (중요)
**파일**: `src/parser/form_parser.py:254-258`

`label in normalized_text or normalized_text in label` 조건이 너무 느슨함:
- `"등록번호"`가 `"최초등록일"`의 `"등록"` 부분과 매칭 가능
- 짧은 값 셀 (예: `"차"`)이 `"차종"` 라벨로 오인식
- `_extract_from_table`의 가드 체크(`not self._match_label(value)`)에서 유효한 값이 라벨로 오분류

**수정 요청**: 최소 길이 조건(2자 이상) 추가 + exact match 우선, substring은 엄격 조건으로

### Issue 2: 싱글턴 초기화 경쟁 조건 (경미)
**파일**: `src/ocr/layout_engine.py:27-41`

`_initialized = True`가 try 블록 밖에 있어 `KeyboardInterrupt` 시 불완전 상태 가능.
현재 Gradio의 동기 처리 환경에서는 실질적 위험 낮음.

**수정 권장**: `finally` 블록으로 이동하거나 `_init_engine()` 성공 후에만 설정

### Issue 3: `get_layout_engine()` 스레드 안전성 (참고)
**파일**: `src/processor.py:44-56`

동시 접근 시 이중 초기화 가능. Gradio의 기본 동작이 동기적이므로 현재는 문제 없으나, 향후 concurrent 처리 시 Lock 필요.

**수정 권장 (장기)**: `threading.Lock()` 가드 추가

## 조건 (CONDITIONAL_PASS)

**필수 수정 (머지 전)**:
1. Issue 1: `_match_label` 서브스트링 매칭 로직 강화

**권장 수정 (머지 후 가능)**:
2. Issue 2: 싱글턴 초기화 안전성
3. Issue 3: 스레드 안전성 (Gradio concurrent 모드 도입 시)

필수 수정 완료 후 Orchestrator에게 머지 승인 요청 가능.
