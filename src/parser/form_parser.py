# -*- coding: utf-8 -*-
"""
Form-aware parser for Korean Vehicle Registration Certificate.
Uses PP-Structure layout analysis results (tables, text regions) to extract fields
based on document structure rather than regex-only parsing.

차량등록증 양식 구조:
  상단: 제목 "자동차등록증"
  기본정보 테이블 (①-⑩): 차량번호, 차종, 용도, 차명, 형식, 차대번호, 원동기, 소유자 등
  제원 테이블: 길이, 너비, 높이, 총중량, 승차정원
  하단: 등록일, 주소 등
"""
import re
import logging

logger = logging.getLogger(__name__)

# 차량등록증 테이블 라벨 → 필드명 매핑
# 라벨은 OCR 변형을 고려하여 여러 패턴 포함
LABEL_FIELD_MAP = {
    'vehicle_no': ['자동차등록번호', '등록번호', '차량번호'],
    'vehicle_type': ['차종', '차 종'],
    'model_name': ['차명', '차 명'],
    'model_year': ['연식', '모델연도', '연 식'],
    'vin': ['차대번호', '차대 번호'],
    'engine_type': ['원동기형식', '원동기 형식', '원동기'],
    'owner_name': ['성명', '명칭', '소유자', '성명(명칭)'],
    'registration_date': ['최초등록일', '등록일', '최초 등록일'],
    'fuel_type': ['연료', '연료의종류', '연료의 종류'],
    'usage': ['용도', '용 도'],
    'length_mm': ['길이', '길 이'],
    'width_mm': ['너비', '너 비'],
    'height_mm': ['높이', '높 이'],
    'total_weight_kg': ['총중량', '총 중량'],
    'passenger_capacity': ['승차정원', '승차 정원', '정원'],
}

# 역방향 매핑: 라벨 텍스트 → 필드명
_REVERSE_MAP = {}
for field, labels in LABEL_FIELD_MAP.items():
    for label in labels:
        _REVERSE_MAP[label] = field


class FormParser:
    """
    Parse PP-Structure layout analysis results into structured vehicle registration fields.
    Works with table HTML cells and text region detections.
    """

    def parse_layout(self, layout_result, img_height=None, img_width=None):
        """
        Extract vehicle registration fields from PP-Structure layout result.

        Args:
            layout_result: dict from LayoutEngine.analyze() with 'tables', 'text_regions'
            img_height: image height in pixels (for zone-based extraction)
            img_width: image width in pixels

        Returns:
            dict: field_name → {'value': str, 'confidence': float, 'source': str}
        """
        fields = {}

        # Strategy 1: Extract from table cells (most reliable for structured forms)
        tables = layout_result.get('tables', [])
        for table in tables:
            table_fields = self._extract_from_table(table)
            self._merge_fields(fields, table_fields, source_name='table')

        # Strategy 2: Extract from text regions with spatial proximity
        text_regions = layout_result.get('text_regions', [])
        if text_regions and img_height and img_width:
            region_fields = self._extract_from_regions(
                text_regions, img_height, img_width
            )
            self._merge_fields(fields, region_fields, source_name='text_region')

        logger.info(
            f"FormParser extracted {len(fields)} fields: "
            f"{[f for f in fields]}"
        )
        return fields

    def _extract_from_table(self, table):
        """
        Extract fields from a single table's cells.
        Looks for label-value pairs in adjacent cells (same row or label above value).
        """
        cells = table.get('cells', [])
        if not cells:
            # Try parsing HTML directly
            html = table.get('html', '')
            if html:
                cells = self._cells_from_html(html)

        if not cells:
            return {}

        fields = {}

        # Build row-based structure
        rows = {}
        for cell in cells:
            row = cell.get('row', 0)
            rows.setdefault(row, []).append(cell)

        # Sort cells within each row by column
        for row_cells in rows.values():
            row_cells.sort(key=lambda c: c.get('col', 0))

        # Strategy A: Adjacent cells in same row (label | value)
        for _, row_cells in sorted(rows.items()):
            for i, cell in enumerate(row_cells):
                cell_text = self._normalize_label(cell.get('text', ''))
                field_name = self._match_label(cell_text)
                if field_name and i + 1 < len(row_cells):
                    value = row_cells[i + 1].get('text', '').strip()
                    if value and not self._match_label(self._normalize_label(value)):
                        fields[field_name] = {
                            'value': self._clean_value(field_name, value),
                            'confidence': 0.85,
                        }

        # Strategy B: Label in one row, value in next row same column
        sorted_rows = sorted(rows.items())
        for idx, (_, row_cells) in enumerate(sorted_rows):
            for cell in row_cells:
                cell_text = self._normalize_label(cell.get('text', ''))
                field_name = self._match_label(cell_text)
                if field_name and field_name not in fields:
                    col = cell.get('col', 0)
                    # Look in next row for same column
                    if idx + 1 < len(sorted_rows):
                        next_row_cells = sorted_rows[idx + 1][1]
                        for nc in next_row_cells:
                            if nc.get('col', -1) == col:
                                value = nc.get('text', '').strip()
                                if value and not self._match_label(
                                    self._normalize_label(value)
                                ):
                                    fields[field_name] = {
                                        'value': self._clean_value(
                                            field_name, value
                                        ),
                                        'confidence': 0.75,
                                    }

        return fields

    def _extract_from_regions(self, text_regions, img_height, img_width):
        """
        Extract fields from text regions using spatial proximity.
        A text region containing a known label → the nearest region to its right
        or below is likely the value.
        """
        fields = {}

        # Classify each region as label or value
        label_regions = []
        value_regions = []

        for region in text_regions:
            text = region.get('text', '').strip()
            if not text:
                continue
            normalized = self._normalize_label(text)
            field_name = self._match_label(normalized)
            if field_name:
                label_regions.append((field_name, region))
            else:
                value_regions.append(region)

        # For each label, find nearest value region to the right or below
        for field_name, label_reg in label_regions:
            if field_name in fields:
                continue

            lx1, ly1, lx2, ly2 = label_reg.get('bbox', (0, 0, 0, 0))
            label_cx = (lx1 + lx2) / 2
            label_cy = (ly1 + ly2) / 2
            label_right = lx2

            best_value = None
            best_dist = float('inf')

            for vreg in value_regions:
                vx1, vy1, vx2, vy2 = vreg.get('bbox', (0, 0, 0, 0))
                vcx = (vx1 + vx2) / 2
                vcy = (vy1 + vy2) / 2

                # Value should be to the right of or below the label
                if vcx < label_cx - img_width * 0.05:
                    continue

                # Prefer same-row (similar Y) then below
                dy = abs(vcy - label_cy)
                dx = vcx - label_right
                if dx < 0:
                    dx = 0

                # Weight: horizontal proximity matters more for same-row
                dist = (dx / img_width) + (dy / img_height) * 2
                if dist < best_dist:
                    best_dist = dist
                    best_value = vreg

            if best_value and best_dist < 0.3:
                value = best_value.get('text', '').strip()
                conf = best_value.get('confidence', 0.5)
                if value:
                    fields[field_name] = {
                        'value': self._clean_value(field_name, value),
                        'confidence': conf * 0.9,  # Slight discount for spatial match
                    }

        return fields

    def _cells_from_html(self, html):
        """Parse HTML table into cell list."""
        cells = []
        rows = re.findall(r'<tr>(.*?)</tr>', html, re.DOTALL)
        for row_idx, row_html in enumerate(rows):
            col_idx = 0
            for cell_match in re.finditer(
                r'<t[dh][^>]*>(.*?)</t[dh]>', row_html, re.DOTALL
            ):
                cell_text = re.sub(r'<[^>]+>', '', cell_match.group(1)).strip()
                cells.append({
                    'row': row_idx,
                    'col': col_idx,
                    'text': cell_text,
                })
                col_idx += 1
        return cells

    @staticmethod
    def _normalize_label(text):
        """Normalize label text: remove spaces, parentheses, circled numbers."""
        text = re.sub(r'[\u2460-\u2469]', '', text)  # Remove ①-⑩
        text = re.sub(r'[()（）\s]', '', text)
        return text.strip()

    @staticmethod
    def _match_label(normalized_text):
        """Match normalized text against known field labels. Returns field name or None."""
        if not normalized_text or len(normalized_text) < 2:
            return None

        # Direct match (highest priority)
        if normalized_text in _REVERSE_MAP:
            return _REVERSE_MAP[normalized_text]

        # Contained-label match: known label is fully contained in the text
        # e.g., "②차종" contains "차종" → match
        # Only match when the label covers most of the text (>=50%) to avoid false positives
        best_match = None
        best_ratio = 0.0
        for label, field in _REVERSE_MAP.items():
            if label in normalized_text:
                ratio = len(label) / len(normalized_text)
                if ratio > best_ratio and ratio >= 0.5:
                    best_ratio = ratio
                    best_match = field

        return best_match

    @staticmethod
    def _clean_value(field_name, value):
        """Clean extracted value based on field type."""
        if not value:
            return value

        # Remove leading/trailing noise
        value = re.sub(r'^[\u2460-\u2469\s:]+', '', value)
        value = value.strip()

        if field_name == 'vin':
            # VIN should be uppercase alphanumeric only
            value = re.sub(r'[^A-Z0-9]', '', value.upper())
        elif field_name in ('length_mm', 'width_mm', 'height_mm', 'total_weight_kg'):
            # Numeric fields: extract digits
            match = re.search(r'(\d[\d,]+)', value)
            if match:
                value = match.group(1).replace(',', '')
        elif field_name == 'passenger_capacity':
            match = re.search(r'(\d+)', value)
            if match:
                value = match.group(1)
        elif field_name == 'registration_date':
            value = re.sub(r'[년월일\s]+', '-', value).strip('-')
        elif field_name == 'model_year':
            match = re.search(r'((?:19|20)\d{2})', value)
            if match:
                value = match.group(1)

        return value

    @staticmethod
    def _merge_fields(target, source, source_name='unknown'):
        """Merge source fields into target. Higher confidence wins."""
        for field, info in source.items():
            if field not in target:
                info['source'] = source_name
                target[field] = info
            elif info.get('confidence', 0) > target[field].get('confidence', 0):
                info['source'] = source_name
                target[field] = info
