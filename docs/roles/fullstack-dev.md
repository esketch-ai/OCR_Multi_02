# Role: Full-stack Developer (풀스택 개발)

## 역할 정의
Orchestrator의 작업 지시서에 따라 기능을 구현하고, QAQC가 검증할 수 있는 상태로 코드를 제출한다.

## 책임 범위
- **기능 구현**: OCR 엔진, 파서, UI, 스토리지 등 전 레이어 개발
- **단위 테스트 작성**: 구현한 기능에 대한 테스트 코드
- **코드 품질**: 컨벤션 준수, 가독성, 성능
- **기술 문서**: 복잡한 로직에 대한 인라인 주석, 필요 시 설계 메모
- **브랜치 관리**: feat/fix 브랜치에서 작업, PR 생성

## 하지 않는 것
- 작업 우선순위 변경 (Orchestrator 역할)
- main 브랜치에 직접 머지 (QAQC 통과 후 Orchestrator 승인)
- 요구사항 범위 임의 확장

## 모듈별 개발 가이드

### OCR 레이어 (`src/ocr/`)
- `paddle_engine.py`: PaddleOCR 초기화/호출. 언어=korean, GPU 설정
- `preprocessor.py`: 이미지 전처리 파이프라인 (그레이스케일, 이진화, 회전 보정)
- 새 OCR 엔진 추가 시: 동일 인터페이스 유지 (텍스트+bbox 반환)

### 파서 레이어 (`src/parser/`)
- `car_registration.py`: OCR 텍스트를 13개 필드로 파싱
- 정규식 + 위치 기반 매칭 혼합 전략
- 새 문서 유형 추가 시: 별도 파서 모듈 생성

### 검증 레이어 (`src/validator/`)
- `standards.py`: 차종/연료 표준값 사전 매칭
- `vin_validator.py`: 차대번호 형식/체크섬 검증
- 검증 실패 = 경고 (에러로 중단하지 않음)

### 스토리지 레이어 (`src/storage/`)
- `excel_writer.py`: 결과 → Excel 변환
- 새 출력 포맷 추가 시: 별도 writer 모듈

### UI 레이어 (`app.py`)
- Gradio 컴포넌트 기반
- HF Spaces 환경 고려 (SSR 비활성화, 파일 크기 제한)

## 작업 흐름
```
1. docs/handoff/ 에서 작업 지시서 확인
2. feat/<name> 또는 fix/<name> 브랜치 생성
3. 구현 + 테스트 작성
4. pytest 통과 확인
5. docs/handoff/ 에 완료 보고서 작성
   - 파일명: YYYY-MM-DD-done-<slug>.md
   - 내용: 변경사항 요약, 테스트 결과, 알려진 제약
6. QAQC 검증 요청
```

## 코딩 규칙
- 함수/변수: snake_case
- 클래스: PascalCase
- 상수: UPPER_SNAKE_CASE
- import 순서: stdlib → third-party → local
- 타입 힌트 권장 (필수는 아님)
- 에러 로깅 시 traceback 포함: `logger.error("msg", exc_info=True)`
- 개별 파일 처리 실패는 catch하고 다음 파일 계속 진행
