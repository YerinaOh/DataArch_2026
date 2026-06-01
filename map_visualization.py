#!/usr/bin/env python3
"""
서대문구 정비 사각지대 — Folium 지도 시각화

입력: seodaemun_result.csv
출력: seodaemun_maintenance_map.html

GeoJSON (행정동 경계) 우선순위:
  1) --geojson 로컬 파일
  2) data/seodaemun_hangjeong.geojson (있으면)
  3) 원격 다운로드 시도
  4) 내장 Mock 폴리곤 생성

공식·공개 GeoJSON 다운로드 참고:
  - 서울 열린데이터광장 행정구역(행정동) 경계:
    https://data.seoul.go.kr/dataList/549/S/1/datasetView.do
  - 통계지리정보(SGIS) 행정동 경계:
    https://sgis.kostat.go.kr/
  - vuski/admdongkor (전국 행정동, adm_cd 필터 11410*):
    https://github.com/vuski/admdongkor
  - southkorea/seoul-maps:
    https://github.com/southkorea/seoul-maps
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import folium
import pandas as pd
import requests
from branca.element import MacroElement
from jinja2 import Template

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_CSV = "seodaemun_result.csv"
OUTPUT_HTML = "seodaemun_maintenance_map.html"
DEFAULT_GEOJSON_PATH = Path("data") / "seodaemun_hangjeong.geojson"

# 서대문구 중심 (충정로 인근)
MAP_CENTER = (37.5790, 126.9365)
MAP_ZOOM = 13

REGION_COLORS: dict[str, str] = {
    "정비사각지대": "#D32F2F",
    "정비중 지역": "#F57C00",
    "잠재쇠퇴지역": "#FBC02D",
    "양호지역": "#388E3C",
}

# Mock용 행정동 대표 좌표 (WGS84) — 실제 GeoJSON 대체용 근사 위치
DONG_CENTROIDS: dict[str, tuple[float, float]] = {
    "충정로2가동": (37.5635, 126.9645),
    "충정로3가동": (37.5665, 126.9680),
    "천연동": (37.5710, 126.9670),
    "신촌동": (37.5555, 126.9360),
    "연희동": (37.5720, 126.9310),
    "홍제1동": (37.5885, 126.9490),
    "홍제2동": (37.5845, 126.9515),
    "홍제3동": (37.5805, 126.9470),
    "홍은1동": (37.5920, 126.9400),
    "홍은2동": (37.5955, 126.9350),
    "남가좌1동": (37.5755, 126.9180),
    "남가좌2동": (37.5705, 126.9120),
    "북가좌1동": (37.5820, 126.9080),
    "북가좌2동": (37.5770, 126.9030),
}

REMOTE_GEOJSON_URLS: list[str] = [
    # 통계청 기반 전국 행정동 (릴리스 자산 경로는 버전마다 다를 수 있음)
    "https://github.com/vuski/admdongkor/releases/latest/download/admdong.geojson",
]

SCORE_COLUMNS = ["obsolescence_score", "decline_score", "development_score"]


def _square_ring(lat: float, lon: float, half_size: float = 0.007) -> list[list[float]]:
    """[lon, lat] GeoJSON Polygon 외곽."""
    return [
        [lon - half_size, lat - half_size],
        [lon + half_size, lat - half_size],
        [lon + half_size, lat + half_size],
        [lon - half_size, lat + half_size],
        [lon - half_size, lat - half_size],
    ]


def generate_mock_geojson(dongs: list[str]) -> dict[str, Any]:
    """행정동별 사각형 Mock 경계 GeoJSON 생성."""
    features: list[dict[str, Any]] = []
    for dong in dongs:
        if dong not in DONG_CENTROIDS:
            logger.warning("Mock 좌표 없음, 건너뜀: %s", dong)
            continue
        lat, lon = DONG_CENTROIDS[dong]
        features.append(
            {
                "type": "Feature",
                "properties": {"dong": dong, "adm_nm": dong, "source": "mock"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [_square_ring(lat, lon)],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _normalize_dong_name(name: str) -> str:
    name = str(name).strip()
    if name.endswith("동") and not name.endswith("가동"):
        return name
    if name.endswith("가") and not name.endswith("가동"):
        return name + "동"
    return name


def _extract_dong_from_properties(props: dict[str, Any]) -> Optional[str]:
    """GeoJSON 속성에서 행정동명 추출."""
    candidates = [
        props.get("dong"),
        props.get("adm_nm"),
        props.get("ADM_NM"),
        props.get("EMD_NM"),
        props.get("emd_nm"),
        props.get("name"),
        props.get("NAME"),
        props.get("adm_nm2"),
    ]
    for val in candidates:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        text = _normalize_dong_name(str(val))
        if text.endswith("동"):
            return text
    # adm_cd: 11410510 -> 서대문구 행정동 코드 (10자리) — 이름 없을 때 스킵
    return None


def filter_seodaemun_features(geojson: dict[str, Any]) -> dict[str, Any]:
    """서대문구(시군구코드 11410) 행정동만 필터."""
    features: list[dict[str, Any]] = []
    for feat in geojson.get("features", []):
        props = feat.get("properties") or {}
        adm_cd = str(props.get("adm_cd") or props.get("ADM_CD") or props.get("SIG_CD") or "")
        name_blob = json.dumps(props, ensure_ascii=False)
        if adm_cd.startswith("11410") or "서대문" in name_blob:
            dong = _extract_dong_from_properties(props)
            if dong:
                props = {**props, "dong": dong}
                feat = {**feat, "properties": props}
            features.append(feat)
    return {"type": "FeatureCollection", "features": features}


def download_geojson(url: str, timeout: int = 120) -> dict[str, Any]:
    logger.info("GeoJSON 다운로드: %s", url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def load_geojson(
    geojson_path: Optional[str | Path] = None,
    allow_mock: bool = True,
) -> tuple[dict[str, Any], str]:
    """
    GeoJSON 로드. 반환: (geojson, source_description)
    """
    paths_to_try: list[Path] = []
    if geojson_path:
        paths_to_try.append(Path(geojson_path))
    paths_to_try.append(DEFAULT_GEOJSON_PATH)

    for path in paths_to_try:
        if path.is_file():
            logger.info("GeoJSON 로컬 로드: %s", path)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            filtered = filter_seodaemun_features(data)
            if filtered["features"]:
                return filtered, f"file:{path}"
            logger.warning("로컬 GeoJSON에서 서대문 행정동을 찾지 못함 — Mock 시도")

    for url in REMOTE_GEOJSON_URLS:
        try:
            data = download_geojson(url)
            filtered = filter_seodaemun_features(data)
            if filtered["features"]:
                DEFAULT_GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(DEFAULT_GEOJSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(filtered, f, ensure_ascii=False)
                logger.info("다운로드 GeoJSON 저장: %s", DEFAULT_GEOJSON_PATH)
                return filtered, f"download:{url}"
        except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
            logger.warning("원격 GeoJSON 실패 (%s): %s", url, exc)

    if not allow_mock:
        raise FileNotFoundError(
            "GeoJSON을 찾을 수 없습니다. --geojson 경로를 지정하거나 "
            f"{DEFAULT_GEOJSON_PATH} 에 파일을 배치하세요."
        )

    logger.info("Mock GeoJSON 생성 (실제 경계는 data.seoul.go.kr 등에서 받아 --geojson 권장)")
    mock = generate_mock_geojson(list(DONG_CENTROIDS.keys()))
    DEFAULT_GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_GEOJSON_PATH, "w", encoding="utf-8") as f:
        json.dump(mock, f, ensure_ascii=False)
    logger.info("Mock GeoJSON 저장: %s", DEFAULT_GEOJSON_PATH)
    return mock, "mock:generated"


def merge_result_with_geojson(
    result_df: pd.DataFrame, geojson: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """결과 CSV와 GeoJSON 병합. 매칭 안 된 동은 Mock 폴리곤 추가."""
    df = result_df.copy()
    if "dong" not in df.columns:
        raise ValueError("seodaemun_result.csv 에 dong 컬럼이 필요합니다.")

    dong_to_row = {row["dong"]: row for _, row in df.iterrows()}
    merged_features: list[dict[str, Any]] = []
    matched_dongs: set[str] = set()

    for feat in geojson.get("features", []):
        props = dict(feat.get("properties") or {})
        dong = props.get("dong") or _extract_dong_from_properties(props)
        if not dong:
            continue
        # CSV 동명과 유연 매칭
        target = dong if dong in dong_to_row else None
        if target is None:
            for csv_dong in dong_to_row:
                if csv_dong in dong or dong in csv_dong:
                    target = csv_dong
                    break
        if target is None:
            continue

        row = dong_to_row[target]
        matched_dongs.add(target)
        enriched = {
            **props,
            "dong": target,
            "region_type": row.get("region_type", ""),
            "obsolescence_score": float(row.get("obsolescence_score", 0)),
            "decline_score": float(row.get("decline_score", 0)),
            "development_score": float(row.get("development_score", 0)),
        }
        merged_features.append(
            {**feat, "properties": enriched}
        )

    missing = set(df["dong"]) - matched_dongs
    if missing:
        logger.warning("GeoJSON 미매칭 동 → Mock 폴리곤 추가: %s", sorted(missing))
        mock = generate_mock_geojson(sorted(missing))
        for feat in mock["features"]:
            dong = feat["properties"]["dong"]
            row = dong_to_row[dong]
            feat["properties"] = {
                **feat["properties"],
                "region_type": row.get("region_type", ""),
                "obsolescence_score": float(row.get("obsolescence_score", 0)),
                "decline_score": float(row.get("decline_score", 0)),
                "development_score": float(row.get("development_score", 0)),
            }
            merged_features.append(feat)

    merged_gj = {"type": "FeatureCollection", "features": merged_features}
    return merged_gj, df


def _style_function(feature: dict[str, Any]) -> dict[str, Any]:
    region = (feature.get("properties") or {}).get("region_type", "")
    color = REGION_COLORS.get(region, "#9E9E9E")
    return {
        "fillColor": color,
        "color": "#424242",
        "weight": 1.5,
        "fillOpacity": 0.65,
    }


def _highlight_function(_feature: dict[str, Any]) -> dict[str, Any]:
    return {"weight": 3, "color": "#212121", "fillOpacity": 0.85}


class Legend(MacroElement):
    """우측 상단 범례."""

    def __init__(self) -> None:
        super().__init__()
        self._name = "Legend"
        items = "".join(
            f'<div style="margin-bottom:4px;">'
            f'<span style="display:inline-block;width:14px;height:14px;'
            f"background:{color};margin-right:6px;border:1px solid #333;\"></span>"
            f"{label}</div>"
            for label, color in REGION_COLORS.items()
        )
        self._template = Template(
            f"""
            {{% macro html(this, kwargs) %}}
            <div style="
                position: fixed;
                top: 12px;
                right: 12px;
                z-index: 9999;
                background: white;
                border: 2px solid #757575;
                border-radius: 6px;
                padding: 10px 12px;
                font-family: 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
                font-size: 13px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.25);
            ">
                <div style="font-weight:bold;margin-bottom:8px;">정비 유형 범례</div>
                {items}
            </div>
            {{% endmacro %}}
            """
        )


def _sanitize_folium_html(html_path: Path) -> None:
    """
    Folium 저장 HTML의 알려진 결함 보정.
    - Python on_each_feature가 JS에 그대로 들어가는 구문 오류
    - 객체 리터럴 spread(...{) — 일부 환경/린터 호환
  """
    text = html_path.read_text(encoding="utf-8")

    # 잘못 직렬화된 Python 콜백 호출 제거
    text = re.sub(
        r"\s*\(<function [^>]+>\)\(feature, layer\);\n",
        "\n",
        text,
    )
    # L.map / L.geoJson 옵션의 빈 spread 제거
    text = re.sub(
        r",\s*\.\.\.\{\s*\n\s*\"zoom\":\s*(\d+),\s*\n\s*\"zoomControl\":\s*(true|false),\s*\n\s*\"preferCanvas\":\s*(true|false),\s*\n\}\s*\n",
        r', "zoom": \1, "zoomControl": \2, "preferCanvas": \3\n',
        text,
    )
    text = re.sub(
        r",\s*\.\.\.\{\s*\n\}\s*\n(\s*\}\);)",
        r"\n\1",
        text,
    )
    # </body> 뒤 script → body 안으로 이동
    body_close = "</body>"
    script_close = "</script>"
    if body_close in text and text.find(body_close) < text.rfind("<script>"):
        idx_body = text.index(body_close)
        idx_script_start = text.index("<script>", idx_body)
        idx_script_end = text.index(script_close, idx_script_start) + len(script_close)
        script_block = text[idx_script_start:idx_script_end]
        text = text[:idx_body] + "\n" + script_block + "\n" + text[idx_body:idx_script_start] + text[idx_script_end:]

    html_path.write_text(text, encoding="utf-8")


def build_map(merged_geojson: dict[str, Any]) -> folium.Map:
    """Choropleth 스타일 GeoJSON + tooltip/popup + 범례."""
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles="OpenStreetMap")

    folium.GeoJson(
        merged_geojson,
        name="서대문구 정비 유형",
        style_function=_style_function,
        highlight_function=_highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "dong",
                "region_type",
                "obsolescence_score",
                "decline_score",
                "development_score",
            ],
            aliases=["행정동", "유형", "노후도", "쇠퇴도", "개발 현황"],
            localize=True,
            sticky=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=[
                "dong",
                "region_type",
                "obsolescence_score",
                "decline_score",
                "development_score",
            ],
            aliases=["행정동", "유형", "노후도", "쇠퇴도", "개발 현황"],
            localize=True,
        ),
    ).add_to(m)

    m.get_root().add_child(Legend())
    folium.LayerControl().add_to(m)
    return m


def run_visualization(
    input_csv: str | Path = INPUT_CSV,
    output_html: str | Path = OUTPUT_HTML,
    geojson_path: Optional[str | Path] = None,
    allow_mock: bool = True,
) -> folium.Map:
    result_df = pd.read_csv(input_csv, encoding="utf-8-sig")
    for col in SCORE_COLUMNS + ["region_type", "dong"]:
        if col not in result_df.columns:
            raise ValueError(f"필수 컬럼 누락: {col}")

    geojson, source = load_geojson(geojson_path, allow_mock=allow_mock)
    logger.info("GeoJSON 출처: %s (%d features)", source, len(geojson.get("features", [])))

    merged_geojson, _ = merge_result_with_geojson(result_df, geojson)
    logger.info("병합된 행정동 수: %d", len(merged_geojson["features"]))

    m = build_map(merged_geojson)
    out = Path(output_html)
    m.save(str(out))
    _sanitize_folium_html(out)
    logger.info("지도 저장: %s", output_html)
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="서대문구 정비 유형 Folium 지도")
    parser.add_argument("-i", "--input", default=INPUT_CSV, help="결과 CSV")
    parser.add_argument("-o", "--output", default=OUTPUT_HTML, help="출력 HTML")
    parser.add_argument(
        "--geojson",
        default=None,
        help=f"행정동 GeoJSON 경로 (미지정 시 {DEFAULT_GEOJSON_PATH} 또는 Mock)",
    )
    parser.add_argument(
        "--no-mock",
        action="store_true",
        help="GeoJSON 없을 때 Mock 생성 금지",
    )
    args = parser.parse_args()

    run_visualization(
        input_csv=args.input,
        output_html=args.output,
        geojson_path=args.geojson,
        allow_mock=not args.no_mock,
    )


if __name__ == "__main__":
    main()
