# -*- coding: utf-8 -*-
"""자동차관리법 기반 차량 규격 표준 상수

자동차관리법 시행규칙 별표1 기준 차종 분류 및 규격 범위.
OCR 값 검증 및 교차 검증에 사용.
Ported from OCR_Vehicle_02.
"""

# ── 차종 (규모 x 유형 조합) ──
VEHICLE_TYPES = {
    "경형승용", "소형승용", "중형승용", "대형승용",
    "경형승합", "소형승합", "중형승합", "대형승합",
    "경형화물", "소형화물", "중형화물", "대형화물",
    "경형특수", "소형특수", "중형특수", "대형특수",
    "소형이륜", "중형이륜", "대형이륜",
}

# ── 용도 ──
USAGE_TYPES = {"자가용", "영업용", "관용"}

# ── 연료타입 ──
FUEL_TYPES = {
    "휘발유", "경유", "LPG", "CNG", "LNG",
    "전기", "수소전기", "수소",
    "하이브리드", "휘발유+전기", "경유+전기", "LPG+전기",
    "플러그인하이브리드", "휘발유+전기(PHEV)", "경유+전기(PHEV)",
    "기타",
}

# OCR 오인식 보정 매핑
FUEL_TYPE_CORRECTIONS = {
    "수소전키": "수소전기",
    "수소전거": "수소전기",
    "수소전기차": "수소전기",
    "수소전": "수소전기",
    "휘발": "휘발유",
    "디젤": "경유",
    "가솔린": "휘발유",
    "엘피지": "LPG",
    "액화석유가스": "LPG",
    "천연가스": "CNG",
    "압축천연가스": "CNG",
}

# ── 잡음 패턴: 안내문구/법률 텍스트 필터링 ──
NOISE_PATTERNS = [
    "말합니다", "증명합니다", "과태료", "위반", "무보험",
    "사유발생일", "변경등록", "말소등록", "검사 시행",
    "자동차등록규칙", "별지", "번호판교부",
    "형사고발", "범칙금", "벌점", "위반시",
    "보험에 가입", "의무보험", "소유권 이전",
    "자동차관리법", "도로교통법", "등록원부",
]

# ── 차종별 규격 범위 (자동차관리법 시행규칙 별표1) ──
DIMENSION_RANGES = {
    "경형승용": {
        "length_mm": (2500, 3700), "width_mm": (1200, 1700),
        "height_mm": (1200, 2100), "weight_kg": (500, 1400),
        "capacity": (2, 5),
    },
    "소형승용": {
        "length_mm": (3500, 4800), "width_mm": (1400, 1800),
        "height_mm": (1200, 2100), "weight_kg": (800, 2100),
        "capacity": (4, 7),
    },
    "중형승용": {
        "length_mm": (4200, 5100), "width_mm": (1650, 1950),
        "height_mm": (1200, 2100), "weight_kg": (1100, 2500),
        "capacity": (5, 7),
    },
    "대형승용": {
        "length_mm": (4600, 5700), "width_mm": (1750, 2200),
        "height_mm": (1200, 2100), "weight_kg": (1400, 3500),
        "capacity": (5, 10),
    },
    "경형승합": {
        "length_mm": (2500, 3700), "width_mm": (1300, 1700),
        "height_mm": (1400, 2100), "weight_kg": (600, 1500),
        "capacity": (6, 11),
    },
    "소형승합": {
        "length_mm": (3600, 4800), "width_mm": (1500, 1800),
        "height_mm": (1700, 2200), "weight_kg": (1000, 3500),
        "capacity": (11, 15),
    },
    "중형승합": {
        "length_mm": (4700, 9200), "width_mm": (1700, 2300),
        "height_mm": (1900, 3300), "weight_kg": (2500, 12000),
        "capacity": (16, 35),
    },
    "대형승합": {
        "length_mm": (9000, 14000), "width_mm": (2300, 2600),
        "height_mm": (2800, 3900), "weight_kg": (8000, 25000),
        "capacity": (36, 80),
    },
    "경형화물": {
        "length_mm": (2500, 3700), "width_mm": (1200, 1700),
        "height_mm": (1200, 2100), "weight_kg": (400, 1500),
        "capacity": (2, 3),
    },
    "소형화물": {
        "length_mm": (3600, 4800), "width_mm": (1400, 1800),
        "height_mm": (1400, 2200), "weight_kg": (1000, 3500),
        "capacity": (2, 6),
    },
    "중형화물": {
        "length_mm": (4700, 9500), "width_mm": (1700, 2400),
        "height_mm": (1800, 3600), "weight_kg": (3500, 10000),
        "capacity": (2, 3),
    },
    "대형화물": {
        "length_mm": (7000, 14000), "width_mm": (2200, 2600),
        "height_mm": (2400, 4200), "weight_kg": (8000, 50000),
        "capacity": (2, 3),
    },
    "소형특수": {
        "length_mm": (3000, 4800), "width_mm": (1400, 1800),
        "height_mm": (1400, 2600), "weight_kg": (1000, 3500),
        "capacity": (2, 3),
    },
    "대형특수": {
        "length_mm": (5000, 16000), "width_mm": (2000, 2600),
        "height_mm": (2000, 4500), "weight_kg": (5000, 50000),
        "capacity": (2, 5),
    },
}

# ── 차명별 표준 규격 (차명 → 고정 스펙) ──
# 차명+차종이 같으면 규격이 동일하므로, OCR 누락/오인식 시 보충·교정에 사용
# 키: 차명 내 포함 키워드 (OCR 변형 대응), 값: 표준 규격
MODEL_SPECS = {
    # ── 현대 대형승합 ──
    "뉴슈퍼에어로시티저상": {
        "vehicle_type": "대형승합", "length_mm": "11090", "width_mm": "2490",
        "height_mm": "3100", "total_weight_kg": "18000", "passenger_capacity": "44",
        "fuel_type": "경유",
    },
    "뉴슈퍼에어로시티초저상": {
        "vehicle_type": "대형승합", "length_mm": "11090", "width_mm": "2490",
        "height_mm": "3100", "total_weight_kg": "18000", "passenger_capacity": "44",
        "fuel_type": "경유",
    },
    "슈퍼에어로시티": {
        "vehicle_type": "대형승합", "length_mm": "11090", "width_mm": "2490",
        "height_mm": "3100", "total_weight_kg": "18000", "passenger_capacity": "44",
        "fuel_type": "경유",
    },
    "일렉시티": {
        "vehicle_type": "대형승합", "length_mm": "11090", "width_mm": "2490",
        "height_mm": "3100", "total_weight_kg": "18000", "passenger_capacity": "44",
        "fuel_type": "전기",
    },
    "일렉시티수소": {
        "vehicle_type": "대형승합", "length_mm": "11090", "width_mm": "2490",
        "height_mm": "3340", "total_weight_kg": "18000", "passenger_capacity": "33",
        "fuel_type": "수소전기",
    },
    "유니버스": {
        "vehicle_type": "대형승합", "length_mm": "12000", "width_mm": "2490",
        "height_mm": "3575", "total_weight_kg": "18000", "passenger_capacity": "45",
        "fuel_type": "경유",
    },
    # ── 현대 중형승합 ──
    "그린시티": {
        "vehicle_type": "중형승합", "length_mm": "9000", "width_mm": "2490",
        "height_mm": "3050", "total_weight_kg": "14000", "passenger_capacity": "33",
        "fuel_type": "경유",
    },
    "카운티": {
        "vehicle_type": "소형승합", "length_mm": "7080", "width_mm": "2040",
        "height_mm": "2755", "total_weight_kg": "8500", "passenger_capacity": "25",
        "fuel_type": "경유",
    },
    # ── 현대 화물 ──
    "마이티": {
        "vehicle_type": "소형화물", "length_mm": "6085", "width_mm": "1995",
        "height_mm": "2420", "total_weight_kg": "8500",
        "fuel_type": "경유",
    },
    "파비스": {
        "vehicle_type": "중형화물", "length_mm": "8590", "width_mm": "2340",
        "height_mm": "2945", "total_weight_kg": "15000",
        "fuel_type": "경유",
    },
    "엑시언트": {
        "vehicle_type": "대형화물", "length_mm": "11550", "width_mm": "2495",
        "height_mm": "3465", "total_weight_kg": "40000",
        "fuel_type": "경유",
    },
    # ── 자일대우 대형승합 ──
    "BS110": {
        "vehicle_type": "대형승합", "length_mm": "11050", "width_mm": "2490",
        "height_mm": "3150", "total_weight_kg": "16500", "passenger_capacity": "43",
        "fuel_type": "경유",
    },
    "BS106": {
        "vehicle_type": "대형승합", "length_mm": "10555", "width_mm": "2490",
        "height_mm": "3090", "total_weight_kg": "16000", "passenger_capacity": "40",
        "fuel_type": "경유",
    },
    "FX120": {
        "vehicle_type": "대형승합", "length_mm": "12000", "width_mm": "2490",
        "height_mm": "3575", "total_weight_kg": "18000", "passenger_capacity": "45",
        "fuel_type": "경유",
    },
    "FX116": {
        "vehicle_type": "대형승합", "length_mm": "11635", "width_mm": "2490",
        "height_mm": "3575", "total_weight_kg": "18000", "passenger_capacity": "43",
        "fuel_type": "경유",
    },
    # ── 자일대우 중형승합 ──
    "레스타": {
        "vehicle_type": "소형승합", "length_mm": "7080", "width_mm": "2040",
        "height_mm": "2705", "total_weight_kg": "8500", "passenger_capacity": "25",
        "fuel_type": "경유",
    },
}


def lookup_model_specs(model_name):
    """차명에서 표준 규격 조회. 키워드 포함 매칭 (OCR 변형 대응).

    Args:
        model_name: OCR로 추출한 차명 문자열

    Returns:
        dict or None: 매칭된 표준 규격 딕셔너리
    """
    if not model_name:
        return None

    # 공백 제거 후 매칭
    clean = model_name.replace(' ', '')

    # 1. 긴 키워드부터 매칭 (초저상 > 저상 > 에어로시티 순)
    for keyword in sorted(MODEL_SPECS.keys(), key=len, reverse=True):
        if keyword in clean:
            return MODEL_SPECS[keyword]

    # 2. 영문 모델명 (대소문자 무시)
    clean_upper = clean.upper()
    for keyword in MODEL_SPECS:
        if keyword.upper() in clean_upper:
            return MODEL_SPECS[keyword]

    return None


# ── 차종 미확인 시 전체 허용 범위 ──
UNIVERSAL_RANGES = {
    "length_mm": (1400, 16000),
    "width_mm": (500, 2600),
    "height_mm": (800, 4500),
    "weight_kg": (70, 50000),
    "capacity": (1, 80),
    "model_year": (1970, 2030),
}
