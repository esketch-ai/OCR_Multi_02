# Task: 자동차등록증 폼 양식 학습 기반 인식 모델 도입

- **날짜**: 2026-03-31
- **우선순위**: P1 (높음)
- **유형**: feature

## 목표
자동차등록증의 고정된 폼 양식(레이아웃)을 학습/활용하여 OCR 인식율을 높인다.
현재 범용 OCR + 정규식 파싱을 **폼 구조 인식 → 필드 단위 추출**로 보강한다.

## 배경
현재 파이프라인의 약점:
1. **범용 OCR**: PaddleOCR는 일반 텍스트 인식기. 차량등록증 양식 구조를 모름
2. **정규식 의존**: `car_registration.py`에서 200줄+ 정규식으로 파싱 → 양식 변형, OCR 오인식에 취약
3. **좌표 하드코딩**: `processor.py`의 bbox 추출이 고정 Y%/X% 존 → 양식 비율 변경 시 실패
4. **한글 라벨 오인식**: PaddleOCR가 한글 라벨을 중국어로 인식하는 사례 다수

자동차등록증은 **정형화된 폼**이므로 구조를 사전에 알면 인식률을 대폭 향상 가능.

## 접근 방식 후보 (3가지)

### Option A: PP-Structure 레이아웃 분석 (권장 - 1차)
PaddlePaddle 생태계 내 **PP-StructureV2** 활용
- 테이블/키-밸류 구조 자동 인식
- 기존 PaddleOCR와 호환, 추가 의존성 최소
- HF Spaces 배포 가능 (CPU 환경)

```
기존: 이미지 → PaddleOCR(텍스트) → 정규식 파싱
개선: 이미지 → PP-Structure(레이아웃+테이블) → 구조화 추출 → 보정
```

**구현 범위:**
- `src/ocr/layout_engine.py` 신규: PP-Structure 래퍼
- `src/parser/form_parser.py` 신규: 폼 구조 기반 필드 매핑
- `src/processor.py` 수정: 레이아웃 엔진을 파이프라인에 통합

### Option B: 템플릿 매칭 + Anchor 기반 (보완 - 2차)
차량등록증의 고정 요소(원문자 ①②③, 테이블 선, 로고)를 앵커로 활용
- OpenCV 템플릿 매칭으로 앵커 위치 검출
- 앵커 기준 상대 좌표로 필드 영역 동적 결정
- 양식 비율 변화에 강건

```
이미지 → 앵커 검출(①②③ 위치) → 상대 좌표 계산 → 필드별 ROI 크롭 → OCR
```

**구현 범위:**
- `src/ocr/template_matcher.py` 신규: 앵커 검출 + ROI 계산
- `Data/templates/` 디렉토리: 앵커 이미지 저장
- `src/processor.py` 수정: 템플릿 매칭 결과로 bbox 보정

### Option C: Document AI (LayoutLM 계열) (장기 - 3차)
텍스트 + 레이아웃 + 이미지를 함께 학습하는 멀티모달 모델
- LayoutLMv3 또는 UDOP 기반 fine-tuning
- 최고 정확도 가능, but GPU 필요 + 학습 데이터 필요
- HF Spaces Free tier에서는 실행 불가 → 별도 API 서버 필요

**현 단계에서는 보류. 데이터 축적 후 검토.**

## 제안 로드맵

```
Phase 1 (즉시): Option A - PP-Structure 레이아웃 분석 도입
  ├─ PP-Structure 엔진 통합
  ├─ 테이블/키-밸류 구조 추출
  └─ 기존 정규식 파서와 결과 병합 (앙상블)

Phase 2 (이후): Option B - 템플릿 매칭 보강
  ├─ 원문자 앵커 검출
  ├─ 동적 ROI 크롭 → 필드별 OCR
  └─ Phase 1 결과와 교차 검증

Phase 3 (장기): Option C - Document AI 검토
  ├─ 학습 데이터 수집/라벨링 체계 구축
  └─ LayoutLM fine-tuning 실험
```

## 아키텍처 변경안

```
현재 파이프라인:
  이미지 → 전처리 → PaddleOCR → 텍스트 파싱 → 검증 → 결과

개선 파이프라인:
  이미지 → 전처리 ─┬→ PaddleOCR (텍스트)      ─┐
                    ├→ PP-Structure (레이아웃)   ─┤→ 결과 병합 → 검증 → 결과
                    └→ 템플릿 매칭 (앵커/ROI)    ─┘
```

**핵심 원칙: 기존 파이프라인 유지 + 새 엔진을 병렬로 추가 + 앙상블**
- 새 엔진 실패 시 기존 결과 fallback
- 두 엔진 결과가 다르면 confidence 기반 선택

## 완료 조건
- [ ] PP-Structure 엔진 래퍼 구현 및 단위 테스트
- [ ] 폼 구조 기반 파서 구현
- [ ] 기존 파이프라인과 앙상블 통합
- [ ] 샘플 이미지 5장 이상으로 인식률 비교 (기존 vs 개선)
- [ ] HF Spaces 환경 호환성 확인 (메모리, 패키지)
- [ ] requirements.txt / packages.txt 업데이트

## 제약 조건
- HF Spaces Free tier: CPU only, 16GB RAM, 50GB 디스크
- 추가 모델 크기: PP-Structure ~150MB 이내 권장
- 기존 PaddleOCR 파이프라인 동작에 영향 없어야 함

## 참고
- PP-Structure 문서: https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/README.md
- 현재 OCR 엔진: `src/ocr/paddle_engine.py` (PaddleOCR 2.x/3.x 호환)
- 현재 파서: `src/parser/car_registration.py` (정규식 + bbox 기반)
- 좌표 추출: `src/processor.py:_extract_bbox_fields()` (고정 Y%/X% 존)
