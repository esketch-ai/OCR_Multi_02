# Done: Phase 1 - PP-Structure 레이아웃 분석 엔진 통합

- **날짜**: 2026-03-31
- **대상 지시서**: `2026-03-31-task-form-aware-model.md`

## 변경사항 요약

### 신규 파일
| 파일 | 설명 |
|------|------|
| `src/ocr/layout_engine.py` | PP-Structure 래퍼 (싱글턴, 2.x/3.x 호환) |
| `src/parser/form_parser.py` | 폼 구조 기반 필드 매핑 (테이블 + 텍스트 영역) |
| `tests/test_layout.py` | 14개 단위 테스트 |

### 수정 파일
| 파일 | 변경 내용 |
|------|-----------|
| `src/config.py` | `ENABLE_LAYOUT_ANALYSIS` 설정 추가 (기본 true, 환경변수 제어) |
| `src/processor.py` | `get_layout_engine()` + `_apply_layout_ensemble()` 추가 |

## 아키텍처
```
이미지 → 전처리 ─┬→ PaddleOCR (텍스트)       → 정규식 파싱 ─┐
                  └→ PP-Structure (레이아웃)   → 폼 파싱     ─┤→ 앙상블 병합 → 검증
                                                              │
                  기존 bbox 추출 ──────────────────────────────┘
```

- 레이아웃 엔진 실패 시 기존 파이프라인만으로 동작 (graceful degradation)
- 앙상블: 텍스트 파서 결과가 빈 필드만 레이아웃 결과로 보충

## 폼 파서 추출 전략
1. **테이블 셀 매칭**: HTML 테이블에서 라벨-값 인접 셀 쌍 추출
2. **행간 매칭**: 라벨이 있는 행의 다음 행 같은 열에서 값 추출
3. **공간 근접도**: 텍스트 영역에서 라벨의 우측/하단 가장 가까운 값 영역 매칭

## 테스트 결과
- 신규 14개 테스트: **전부 통과**
- 기존 테스트: 변경 없음 (기존 3개 실패는 이전부터 존재)
- PP-Structure 미설치 환경: graceful fallback 확인

## 알려진 제약

### 로컬 환경 (paddleocr 3.x)
- PPStructureV3는 `paddlex[ocr]` 추가 의존성 필요
- `pip install "paddlex[ocr]"` 실행 필요

### HF Spaces 환경 (paddleocr 2.9.1)
- PPStructure (v2 API) 사용됨 — paddleocr 2.9.1에 번들
- 추가 설치 없이 동작 예상
- **실제 HF Spaces 배포 후 테스트 필요**

### 미완료 (QAQC에서 확인 필요)
- [ ] 실제 차량등록증 이미지로 레이아웃 분석 결과 확인
- [ ] HF Spaces 배포 후 메모리/시간 성능 측정
- [ ] requirements.txt에 paddlex 추가 여부 판단 (로컬 환경용)

## QAQC 검증 요청
QAQC에서 아래 항목 검증 부탁드립니다:
1. 기존 파이프라인 회귀 테스트 (레이아웃 비활성화 상태)
2. 레이아웃 엔진 활성화 시 실제 이미지 인식률 비교
3. HF Spaces 호환성 확인
