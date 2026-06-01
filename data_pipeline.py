#!/usr/bin/env python3
"""
서대문구 정비 사각지대 탐지 — 1단계 데이터 수집·전처리 파이프라인

법정동/행정동 단위로 노후도·쇠퇴도·개발 현황 점수(0~5)를 산정하고
seodaemun_clean_data.csv 로 저장합니다.

환경변수 (.env 지원):
  DATA_GO_KR_API_KEY      건축HUB 건축물대장 (공공데이터포털)
  SEOUL_OPEN_API_KEY      서울 열린데이터광장
  VWORLD_API_KEY          V-World (경관 보조 지표)
  VWORLD_DOMAIN           V-World 신청 도메인 (기본 localhost)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 서대문구 행정동 (14개)
# ---------------------------------------------------------------------------
SEODAEMUN_ADMIN_DONGS: list[str] = [
    "충정로2가동",
    "충정로3가동",
    "천연동",
    "신촌동",
    "연희동",
    "홍제1동",
    "홍제2동",
    "홍제3동",
    "홍은1동",
    "홍은2동",
    "남가좌1동",
    "남가좌2동",
    "북가좌1동",
    "북가좌2동",
]

# 법정동코드(읍면동 5자리) — 시군구 11410 + bjdongCd
SEODAEMUN_BJDONG: list[tuple[str, str, str]] = [
    ("10100", "충정로2가", "충정로2가동"),
    ("10200", "충정로3가", "충정로3가동"),
    ("10300", "합동", "충정로3가동"),
    ("10400", "미근동", "천연동"),
    ("10500", "냉천동", "천연동"),
    ("10600", "천연동", "천연동"),
    ("10700", "옥천동", "천연동"),
    ("10800", "영천동", "천연동"),
    ("10900", "현저동", "연희동"),
    ("11000", "북아현동", "충정로2가동"),
    ("11100", "홍제동", "홍제2동"),
    ("11200", "대현동", "홍제2동"),
    ("11300", "대신동", "홍제1동"),
    ("11400", "신촌동", "신촌동"),
    ("11500", "봉원동", "신촌동"),
    ("11600", "창천동", "신촌동"),
    ("11700", "연희동", "연희동"),
    ("11800", "홍은동", "홍은1동"),
    ("11900", "북가좌동", "북가좌1동"),
    ("12000", "남가좌동", "남가좌1동"),
]

# 홍제·홍은·가좌 등은 행정동이 분할되어 있으므로 API 집계 후 분배
ADMIN_DONG_SPLIT: dict[str, list[str]] = {
    "홍제2동": ["홍제1동", "홍제2동", "홍제3동"],
    "홍은1동": ["홍은1동", "홍은2동"],
    "남가좌1동": ["남가좌1동", "남가좌2동"],
    "북가좌1동": ["북가좌1동", "북가좌2동"],
}

SIGUNGU_CODE = "11410"
OUTPUT_CSV = "seodaemun_clean_data.csv"
SEOUL_PAGE_SIZE = 1000
# CleanupBussinessProgress 에서 BIZ_NO 가 11410- 인 레코드는 약 15,000건 이후 구간에 집중
SEOUL_REDEV_PAGE_START = 15_001
BUILDING_PAGE_SIZE = 100
BUILDING_REQUEST_DELAY_SEC = 0.15

RESIDENTIAL_PURPS_PATTERN = re.compile(
    r"주택|주거|아파트|단독|다세대|연립|공동주택|도시형생활"
)

REDEVELOPMENT_STAGE_SCORE: dict[str, float] = {
    "미추진": 0.0,
    "기본계획": 0.5,
    "계획수립": 1.0,
    "재정비촉진": 1.2,
    "안전진단": 1.3,
    "추진위원회": 1.5,
    "정비구역": 2.0,
    "정비구역지정": 2.0,
    "조합설립": 2.5,
    "조합설립인가": 2.5,
    "사업시행": 3.5,
    "사업시행인가": 3.5,
    "관리처분": 4.0,
    "관리처분인가": 4.0,
    "철거": 4.2,
    "착공": 4.5,
    "분양": 4.6,
    "준공": 5.0,
    "준공인가": 5.0,
    "이전고시": 5.0,
    "완료": 5.0,
    "진행중": 4.0,
}

# 정비사업 TTL 키워드 → 행정동 (복수 매칭 가능)
TTL_DONG_HINTS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"충정로\s*2|충정로2"), ["충정로2가동"]),
    (re.compile(r"충정로\s*3|충정로3"), ["충정로3가동"]),
    (re.compile(r"충정로\s*1|충정로1"), ["충정로2가동", "충정로3가동"]),
    (re.compile(r"천연|냉천|옥천|영천|미근"), ["천연동"]),
    (re.compile(r"신촌|봉원|창천"), ["신촌동"]),
    (re.compile(r"연희"), ["연희동"]),
    (re.compile(r"홍제"), ["홍제1동", "홍제2동", "홍제3동"]),
    (re.compile(r"홍은"), ["홍은1동", "홍은2동"]),
    (re.compile(r"북가좌|북 가좌"), ["북가좌1동", "북가좌2동"]),
    (re.compile(r"남가좌|남 가좌"), ["남가좌1동", "남가좌2동"]),
    (re.compile(r"가좌"), ["남가좌1동", "남가좌2동", "북가좌1동", "북가좌2동"]),
    (re.compile(r"북아현"), ["충정로2가동"]),
]


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
@dataclass
class ApiConfig:
    data_go_kr_key: str = ""
    seoul_open_key: str = ""
    vworld_key: str = ""
    vworld_domain: str = "localhost"

    @classmethod
    def from_environ(cls) -> "ApiConfig":
        return cls(
            data_go_kr_key=os.environ.get("DATA_GO_KR_API_KEY", "").strip(),
            seoul_open_key=os.environ.get("SEOUL_OPEN_API_KEY", "").strip(),
            vworld_key=os.environ.get("VWORLD_API_KEY", "").strip(),
            vworld_domain=os.environ.get("VWORLD_DOMAIN", "localhost").strip() or "localhost",
        )


@dataclass
class DataSourceLog:
    building: str = "missing"
    vacant: str = "missing"
    landscape: str = "missing"
    redevelopment: str = "missing"


# ---------------------------------------------------------------------------
# 점수 유틸
# ---------------------------------------------------------------------------
def normalize_minmax(
    series: pd.Series, target_min: float = 0.0, target_max: float = 5.0
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(target_max / 2.0, index=series.index)
    scaled = (s - lo) / (hi - lo)
    return scaled * (target_max - target_min) + target_min


def weighted_score(parts: list[tuple[pd.Series, float]]) -> pd.Series:
    total_w = sum(w for _, w in parts)
    if total_w == 0:
        raise ValueError("가중치 합이 0입니다.")
    return sum(s * w for s, w in parts) / total_w


def split_admin_dong_rows(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """홍제·홍은·가좌 등 중간 집계 동을 분할 행정동으로 균등 배분."""
    if df.empty:
        return df
    count_cols = {"total_units", "vacant_units", "project_count"}
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        dong = row["dong"]
        targets = ADMIN_DONG_SPLIT.get(dong, [dong])
        if dong not in SEODAEMUN_ADMIN_DONGS and dong not in ADMIN_DONG_SPLIT:
            continue
        for target in targets:
            if target not in SEODAEMUN_ADMIN_DONGS:
                continue
            new_row: dict[str, Any] = {"dong": target}
            for col in value_cols:
                val = row.get(col)
                if col in count_cols and pd.notna(val):
                    new_row[col] = float(val) / len(targets)
                else:
                    new_row[col] = val
            rows.append(new_row)
    if not rows:
        return df[df["dong"].isin(SEODAEMUN_ADMIN_DONGS)].copy()

    out = pd.DataFrame(rows)
    agg_spec: dict[str, str] = {}
    for col in value_cols:
        if col not in out.columns:
            continue
        agg_spec[col] = "sum" if col in count_cols else "mean"
    if not agg_spec:
        return out
    return out.groupby("dong", as_index=False).agg(agg_spec)


def expand_buildings_to_split_dongs(df: pd.DataFrame) -> pd.DataFrame:
    """건축물 레코드를 분할 행정동 목록으로 복제."""
    if df.empty:
        return df
    parts: list[pd.DataFrame] = []
    for dong, grp in df.groupby("dong"):
        targets = ADMIN_DONG_SPLIT.get(dong, [dong])
        valid = [t for t in targets if t in SEODAEMUN_ADMIN_DONGS]
        if not valid:
            continue
        chunk = grp.copy()
        if len(valid) > 1:
            chunk = pd.concat([chunk.assign(dong=t) for t in valid], ignore_index=True)
        else:
            chunk["dong"] = valid[0]
        parts.append(chunk)
    if not parts:
        return df
    return pd.concat(parts, ignore_index=True)


def parse_address_to_admin_dong(address: str) -> Optional[str]:
    if not address:
        return None
    for admin in SEODAEMUN_ADMIN_DONGS:
        short = admin.replace("동", "")
        if admin in address or (short and short in address):
            return admin
    for _, legal, admin in SEODAEMUN_BJDONG:
        if legal in address:
            return admin
    return None


# ---------------------------------------------------------------------------
# 공공데이터 API 공통
# ---------------------------------------------------------------------------
def _api_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    header = payload.get("response", {}).get("header", {})
    code = header.get("resultCode", "")
    if code not in ("00", "0", "000", ""):
        msg = header.get("resultMsg", "")
        logger.warning("API 결과 코드 %s: %s", code, msg)
    body = payload.get("response", {}).get("body", {})
    items = body.get("items", {})
    if not items:
        return []
    item_list = items.get("item", [])
    if isinstance(item_list, dict):
        return [item_list]
    return item_list or []


# ---------------------------------------------------------------------------
# 1. 건축물대장 (건축HUB — getBrTitleInfo)
# ---------------------------------------------------------------------------
class BuildingRegisterClient:
    """
    국토교통부 건축HUB 건축물대장 — 표제부 조회
    https://www.data.go.kr/data/15134735/openapi.do
    """

    BASE_URL = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo"

    def __init__(self, api_key: str, session: Optional[requests.Session] = None):
        self.api_key = api_key
        self.session = session or requests.Session()

    def fetch_page(
        self, bjdong_cd: str, page_no: int = 1, num_of_rows: int = BUILDING_PAGE_SIZE
    ) -> dict[str, Any]:
        params = {
            "serviceKey": self.api_key,
            "sigunguCd": SIGUNGU_CODE,
            "bjdongCd": bjdong_cd,
            "numOfRows": num_of_rows,
            "pageNo": page_no,
            "_type": "json",
        }
        resp = self.session.get(self.BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def parse_buildings(payload: dict[str, Any], default_admin_dong: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        current_year = pd.Timestamp.now().year
        for item in _api_items(payload):
            address = item.get("newPlatPlc") or item.get("platPlc") or ""
            dong = parse_address_to_admin_dong(str(address)) or default_admin_dong
            use_apr = item.get("useAprDay") or ""
            age = np.nan
            if use_apr and str(use_apr)[:4].isdigit():
                age = current_year - int(str(use_apr)[:4])
            purps = str(item.get("mainPurpsCdNm") or "")
            is_residential = bool(RESIDENTIAL_PURPS_PATTERN.search(purps))
            rows.append(
                {
                    "dong": dong,
                    "building_age": age,
                    "is_residential": is_residential,
                    "main_purps": purps,
                }
            )
        return pd.DataFrame(rows)

    def fetch_bjdong(self, bjdong_cd: str, default_admin_dong: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        page = 1
        while True:
            try:
                payload = self.fetch_page(bjdong_cd, page_no=page)
            except requests.RequestException as exc:
                logger.error("건축물대장 [%s] 호출 실패: %s", bjdong_cd, exc)
                break
            df_page = self.parse_buildings(payload, default_admin_dong)
            if df_page.empty:
                break
            frames.append(df_page)
            total = int(payload.get("response", {}).get("body", {}).get("totalCount") or 0)
            if page * BUILDING_PAGE_SIZE >= total:
                break
            page += 1
            time.sleep(BUILDING_REQUEST_DELAY_SEC)
        if not frames:
            return pd.DataFrame(columns=["dong", "building_age", "is_residential", "main_purps"])
        return pd.concat(frames, ignore_index=True)

    def fetch_all_for_seodaemun(self) -> pd.DataFrame:
        if not self.api_key:
            logger.warning("DATA_GO_KR_API_KEY 미설정")
            return pd.DataFrame(columns=["dong", "building_age", "is_residential", "main_purps"])

        frames: list[pd.DataFrame] = []
        for bjdong_cd, _, admin_dong in SEODAEMUN_BJDONG:
            logger.info("건축물대장 수집: bjdongCd=%s (%s)", bjdong_cd, admin_dong)
            df = self.fetch_bjdong(bjdong_cd, admin_dong)
            if not df.empty:
                frames.append(df)
            time.sleep(BUILDING_REQUEST_DELAY_SEC)

        if not frames:
            return pd.DataFrame(columns=["dong", "building_age", "is_residential", "main_purps"])
        combined = pd.concat(frames, ignore_index=True)
        return expand_buildings_to_split_dongs(combined)


def aggregate_building_age_by_dong(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["dong", "avg_building_age", "ratio_over_30y"])

    grouped = raw.groupby("dong", as_index=False).agg(
        avg_building_age=("building_age", "mean"),
        ratio_over_30y=("building_age", lambda s: (pd.to_numeric(s, errors="coerce") >= 30).mean()),
    )
    return grouped


def aggregate_vacant_proxy_from_buildings(raw: pd.DataFrame) -> pd.DataFrame:
    """
    서울시 동별 빈집 Open API 가 없을 때,
    건축물대장 기반 주거용 노후 건축물 비율을 빈집 지표 대용으로 사용.
    """
    if raw.empty:
        return pd.DataFrame(columns=["dong", "total_units", "vacant_units"])

    records = []
    for dong, grp in raw.groupby("dong"):
        res = grp[grp["is_residential"] == True]  # noqa: E712
        total = len(res)
        if total == 0:
            total = len(grp)
            vacant = (pd.to_numeric(grp["building_age"], errors="coerce") >= 30).sum()
        else:
            vacant = (pd.to_numeric(res["building_age"], errors="coerce") >= 30).sum()
        records.append(
            {"dong": dong, "total_units": max(total, 1), "vacant_units": int(vacant)}
        )
    return pd.DataFrame(records)


def score_obsolescence(agg: pd.DataFrame, dongs: list[str]) -> pd.DataFrame:
    base = pd.DataFrame({"dong": dongs})
    merged = base.merge(agg, on="dong", how="left")
    merged["avg_building_age"] = merged["avg_building_age"].fillna(merged["avg_building_age"].median())
    merged["ratio_over_30y"] = merged["ratio_over_30y"].fillna(merged["ratio_over_30y"].median())
    age_norm = normalize_minmax(merged["avg_building_age"])
    ratio_norm = normalize_minmax(merged["ratio_over_30y"])
    merged["obsolescence_score"] = weighted_score([(age_norm, 0.4), (ratio_norm, 0.6)])
    return merged[["dong", "obsolescence_score", "avg_building_age", "ratio_over_30y"]]


# ---------------------------------------------------------------------------
# 2. 서울 열린데이터 + V-World
# ---------------------------------------------------------------------------
class SeoulOpenDataClient:
    """서울 열린데이터광장 Open API."""

    BASE = "http://openapi.seoul.go.kr:8088"
    REDEVELOPMENT_DATASET = "CleanupBussinessProgress"
    SEODAEMUN_BIZ_PREFIX = "11410-"

    def __init__(self, api_key: str, session: Optional[requests.Session] = None):
        self.api_key = api_key
        self.session = session or requests.Session()

    def _get_json(self, dataset: str, start: int, end: int) -> dict[str, Any]:
        url = f"{self.BASE}/{self.api_key}/json/{dataset}/{start}/{end}/"
        resp = self.session.get(url, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        if "RESULT" in data and data["RESULT"].get("CODE", "").startswith("ERROR"):
            raise RuntimeError(data["RESULT"].get("MESSAGE", "Seoul API error"))
        return data

    def fetch_all_rows(self, dataset: str, page_start: int = 1) -> list[dict[str, Any]]:
        first = self._get_json(dataset, 1, 1)
        root = first.get(dataset, {})
        total = int(root.get("list_total_count") or 0)
        if total == 0:
            return []
        all_rows: list[dict[str, Any]] = []
        start_at = max(1, min(page_start, total))
        for start in range(start_at, total + 1, SEOUL_PAGE_SIZE):
            end = min(start + SEOUL_PAGE_SIZE - 1, total)
            chunk = self._get_json(dataset, start, end)
            rows = chunk.get(dataset, {}).get("row", [])
            if isinstance(rows, dict):
                rows = [rows]
            all_rows.extend(rows)
            logger.info("서울 API %s: %d/%d건", dataset, end, total)
            time.sleep(0.2)
        return all_rows

    def fetch_redevelopment_projects(self) -> pd.DataFrame:
        if not self.api_key:
            logger.warning("SEOUL_OPEN_API_KEY 미설정")
            return pd.DataFrame(columns=["dong", "stage", "project_count"])

        try:
            rows = self.fetch_all_rows(
                self.REDEVELOPMENT_DATASET,
                page_start=SEOUL_REDEV_PAGE_START,
            )
        except (requests.RequestException, RuntimeError) as exc:
            logger.error("정비사업 API 실패: %s", exc)
            return pd.DataFrame(columns=["dong", "stage", "project_count"])

        sdm_rows = [
            r
            for r in rows
            if str(r.get("BIZ_NO", "")).startswith(self.SEODAEMUN_BIZ_PREFIX)
        ]
        if not sdm_rows:
            logger.warning("서대문구 정비사업(BIZ_NO 11410-) 데이터 없음")
            return pd.DataFrame(columns=["dong", "stage", "project_count"])

        return self._progress_rows_to_projects(sdm_rows)

    @staticmethod
    def _infer_dongs_from_text(text: str) -> list[str]:
        text = text or ""
        matched: list[str] = []
        for pattern, dongs in TTL_DONG_HINTS:
            if pattern.search(text):
                matched.extend(dongs)
        if not matched:
            return []
        seen: set[str] = set()
        out: list[str] = []
        for d in matched:
            if d not in seen:
                seen.add(d)
                out.append(d)
        return out

    def _progress_rows_to_projects(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        by_biz: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_biz.setdefault(str(row.get("BIZ_NO", "")), []).append(row)

        records: list[dict[str, Any]] = []
        for biz_no, events in by_biz.items():
            best_score = -1.0
            best_stage = "미추진"
            texts: list[str] = []
            for ev in events:
                stage_nm = str(ev.get("SE_NM") or "")
                texts.append(stage_nm)
                texts.append(str(ev.get("TTL") or ""))
                texts.append(str(ev.get("DTL_CN") or ""))
                score = map_stage_to_score(stage_nm)
                if score > best_score:
                    best_score = score
                    best_stage = stage_nm

            hint_text = " ".join(texts)
            dongs = self._infer_dongs_from_text(hint_text)
            if not dongs:
                logger.debug("동 매칭 실패 BIZ_NO=%s TTL=%s", biz_no, hint_text[:80])
                continue

            for dong in dongs:
                records.append(
                    {
                        "dong": dong,
                        "stage": best_stage,
                        "project_count": 1 / len(dongs),
                        "biz_no": biz_no,
                    }
                )

        if not records:
            return pd.DataFrame(columns=["dong", "stage", "project_count"])
        df = pd.DataFrame(records)
        agg = df.groupby("dong", as_index=False).agg(
            stage=("stage", lambda s: max(s, key=map_stage_to_score)),
            project_count=("project_count", "sum"),
        )
        return agg


class VWorldLandscapeClient:
    """
    V-World 2D Data API + 건축물대장 파생 지표로 경관 쇠퇴도 산출.
    geomFilter: 서대문구 bbox → 토지이용(LT_C_UQ111)·경관지구(LT_C_LHBLPN)
    """

    DATA_URL = "https://api.vworld.kr/req/data"
    SEODAEMUN_BBOX = "BOX(126.9136,37.5638,126.9598,37.5824)"

    def __init__(
        self,
        api_key: str,
        domain: str = "localhost",
        session: Optional[requests.Session] = None,
    ):
        self.api_key = api_key
        self.domain = domain
        self.session = session or requests.Session()

    def _get_features(self, data_layer: str, size: int = 1000) -> list[dict[str, Any]]:
        params = {
            "service": "data",
            "request": "GetFeature",
            "data": data_layer,
            "key": self.api_key,
            "domain": self.domain,
            "format": "json",
            "geomFilter": self.SEODAEMUN_BBOX,
            "size": size,
            "geometry": "false",
        }
        resp = self.session.get(self.DATA_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        status = payload.get("response", {}).get("status")
        if status != "OK":
            err = payload.get("response", {}).get("error", {})
            raise RuntimeError(err.get("text", f"V-World status={status}"))
        features = payload["response"]["result"]["featureCollection"]["features"]
        return [f.get("properties") or {} for f in features]

    def fetch_landscape_decline(
        self, dongs: list[str], building_raw: pd.DataFrame
    ) -> pd.DataFrame:
        base = pd.DataFrame({"dong": dongs})
        building_metric = self._building_landscape_proxy(building_raw, dongs)
        if not self.api_key:
            logger.info("VWORLD_API_KEY 미설정 — 건축물·토지이용 파생 지표만 사용")
            return building_metric

        try:
            uq = self._get_features("LT_C_UQ111", size=1000)
            lh = self._get_features("LT_C_LHBLPN", size=100)
            vworld_score = self._score_from_land_use(uq, lh, dongs).rename(
                columns={"landscape_decline": "landscape_vw"}
            )
            merged = base.merge(building_metric, on="dong").merge(
                vworld_score, on="dong", how="left"
            )
            merged["landscape_decline"] = merged[
                ["landscape_decline", "landscape_vw"]
            ].mean(axis=1)
            return merged[["dong", "landscape_decline"]]
        except (requests.RequestException, RuntimeError, KeyError) as exc:
            logger.warning("V-World 경관 API 실패, 건축물 파생 지표 사용: %s", exc)
            return building_metric

    @staticmethod
    def _building_landscape_proxy(building_raw: pd.DataFrame, dongs: list[str]) -> pd.DataFrame:
        if building_raw.empty:
            return pd.DataFrame({"dong": dongs, "landscape_decline": 0.5})

        records = []
        for dong, grp in building_raw.groupby("dong"):
            non_res = float((grp["is_residential"] == False).mean()) if len(grp) else 0.0  # noqa: E712
            old = float((pd.to_numeric(grp["building_age"], errors="coerce") >= 40).mean())
            records.append(
                {
                    "dong": dong,
                    "landscape_decline": 0.6 * non_res + 0.4 * old,
                }
            )
        agg = pd.DataFrame(records)
        base = pd.DataFrame({"dong": dongs})
        return base.merge(agg[["dong", "landscape_decline"]], on="dong", how="left").fillna(
            {"landscape_decline": 0.5}
        )

    @staticmethod
    def _score_from_land_use(
        uq_props: list[dict[str, Any]],
        lh_props: list[dict[str, Any]],
        dongs: list[str],
    ) -> pd.DataFrame:
        non_residential = sum(
            1
            for p in uq_props
            if "주거" not in str(p.get("uname", "")) and "주택" not in str(p.get("uname", ""))
        )
        total_uq = max(len(uq_props), 1)
        gu_non_res_share = non_residential / total_uq

        dong_hits: dict[str, float] = {d: 0.0 for d in dongs}
        for p in lh_props:
            name = str(p.get("zonename", ""))
            for dong in dongs:
                if dong.replace("동", "") in name:
                    dong_hits[dong] += 1.0

        records = []
        for dong in dongs:
            lh_score = min(dong_hits[dong] / 3.0, 1.0)
            decline = 0.5 * gu_non_res_share + 0.5 * lh_score
            records.append({"dong": dong, "landscape_decline": decline})
        return pd.DataFrame(records)


def score_decline(
    vacant_df: pd.DataFrame,
    landscape_df: pd.DataFrame,
    dongs: list[str],
) -> pd.DataFrame:
    base = pd.DataFrame({"dong": dongs})
    if not vacant_df.empty:
        vac = vacant_df.groupby("dong", as_index=False).agg(
            total_units=("total_units", "sum"),
            vacant_units=("vacant_units", "sum"),
        )
        vac["vacant_ratio"] = vac["vacant_units"] / vac["total_units"].replace(0, np.nan)
    else:
        vac = pd.DataFrame(columns=["dong", "vacant_ratio"])

    merged = base.merge(vac[["dong", "vacant_ratio"]], on="dong", how="left")
    merged = merged.merge(landscape_df, on="dong", how="left")
    merged["vacant_ratio"] = merged["vacant_ratio"].fillna(merged["vacant_ratio"].median())
    merged["landscape_decline"] = merged["landscape_decline"].fillna(
        merged["landscape_decline"].median()
    )
    vac_norm = normalize_minmax(merged["vacant_ratio"])
    land_norm = normalize_minmax(merged["landscape_decline"])
    merged["decline_score"] = weighted_score([(vac_norm, 0.5), (land_norm, 0.5)])
    return merged[["dong", "decline_score", "vacant_ratio", "landscape_decline"]]


# ---------------------------------------------------------------------------
# 3. 정비사업
# ---------------------------------------------------------------------------
def map_stage_to_score(stage: str) -> float:
    stage = str(stage).strip()
    for keyword, score in sorted(
        REDEVELOPMENT_STAGE_SCORE.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if keyword in stage:
            return score
    return 0.0


def score_development(redev_df: pd.DataFrame, dongs: list[str]) -> pd.DataFrame:
    base = pd.DataFrame({"dong": dongs})
    if redev_df.empty:
        merged = base.copy()
        merged["stage_raw_score"] = 0.0
        merged["project_count"] = 0
    else:
        redev = redev_df.copy()
        redev["stage_raw_score"] = redev["stage"].map(map_stage_to_score)
        agg = redev.groupby("dong", as_index=False).agg(
            stage_raw_score=("stage_raw_score", "max"),
            project_count=("project_count", "sum"),
        )
        merged = base.merge(agg, on="dong", how="left")
        merged["stage_raw_score"] = merged["stage_raw_score"].fillna(0.0)
        merged["project_count"] = merged["project_count"].fillna(0)

    merged["development_score"] = normalize_minmax(merged["stage_raw_score"])
    return merged[["dong", "development_score", "stage_raw_score", "project_count"]]


# ---------------------------------------------------------------------------
# Mock (테스트)
# ---------------------------------------------------------------------------
def generate_mock_building_data(dongs: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for dong in dongs:
        n = int(rng.integers(80, 200))
        for age in rng.integers(5, 65, size=n):
            purps = "공동주택" if rng.random() > 0.3 else "업무시설"
            rows.append(
                {
                    "dong": dong,
                    "building_age": int(age),
                    "is_residential": "주택" in purps or "주거" in purps,
                    "main_purps": purps,
                }
            )
    return pd.DataFrame(rows)


def generate_mock_vacant_data(dongs: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        [
            {
                "dong": d,
                "total_units": int(rng.integers(500, 3000)),
                "vacant_units": int(rng.integers(20, 400)),
            }
            for d in dongs
        ]
    )


def generate_mock_redevelopment_data(dongs: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    stages = list(REDEVELOPMENT_STAGE_SCORE.keys())
    return pd.DataFrame(
        [
            {
                "dong": d,
                "stage": rng.choice(stages),
                "project_count": float(rng.integers(0, 4)),
            }
            for d in dongs
        ]
    )


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------
def run_pipeline(
    use_mock: bool = False,
    allow_mock_fallback: bool = True,
    output_path: str = OUTPUT_CSV,
) -> pd.DataFrame:
    dongs = SEODAEMUN_ADMIN_DONGS
    config = ApiConfig.from_environ()
    sources = DataSourceLog()

    if use_mock:
        logger.info("Mock 모드")
        building_raw = generate_mock_building_data(dongs)
        vacant_raw = generate_mock_vacant_data(dongs)
        landscape_raw = VWorldLandscapeClient._building_landscape_proxy(building_raw, dongs)
        redev_raw = generate_mock_redevelopment_data(dongs)
        sources = DataSourceLog("mock", "mock", "mock", "mock")
    else:
        bld_client = BuildingRegisterClient(config.data_go_kr_key)
        building_raw = bld_client.fetch_all_for_seodaemun()
        if building_raw.empty:
            if not allow_mock_fallback:
                raise RuntimeError("건축물대장 API 데이터를 가져오지 못했습니다.")
            logger.warning("건축물대장 실패 → Mock")
            building_raw = generate_mock_building_data(dongs)
            sources.building = "mock"
        else:
            sources.building = "api:BldRgstHub/getBrTitleInfo"
            logger.info("건축물 %d건 수집", len(building_raw))

        vacant_raw = aggregate_vacant_proxy_from_buildings(building_raw)
        vacant_raw = split_admin_dong_rows(vacant_raw, ["total_units", "vacant_units"])
        sources.vacant = "proxy:building_register_residential_30y"

        vworld = VWorldLandscapeClient(config.vworld_key, config.vworld_domain)
        landscape_raw = vworld.fetch_landscape_decline(dongs, building_raw)
        sources.landscape = (
            "api:vworld+building" if config.vworld_key else "proxy:building_register"
        )

        seoul = SeoulOpenDataClient(config.seoul_open_key)
        redev_raw = seoul.fetch_redevelopment_projects()
        if redev_raw.empty:
            if not allow_mock_fallback:
                raise RuntimeError("정비사업 API 데이터를 가져오지 못했습니다.")
            logger.warning("정비사업 실패 → Mock")
            redev_raw = generate_mock_redevelopment_data(dongs)
            sources.redevelopment = "mock"
        else:
            sources.redevelopment = "api:CleanupBussinessProgress"
            logger.info("정비사업 %d개 동 매핑", len(redev_raw))

    bld_agg = aggregate_building_age_by_dong(building_raw)
    obsolescence = score_obsolescence(bld_agg, dongs)
    decline = score_decline(vacant_raw, landscape_raw, dongs)
    development = score_development(redev_raw, dongs)

    result = obsolescence.merge(decline, on="dong").merge(development, on="dong")
    result["data_source_building"] = sources.building
    result["data_source_vacant"] = sources.vacant
    result["data_source_landscape"] = sources.landscape
    result["data_source_redevelopment"] = sources.redevelopment

    output_cols = [
        "dong",
        "obsolescence_score",
        "decline_score",
        "development_score",
        "avg_building_age",
        "ratio_over_30y",
        "vacant_ratio",
        "landscape_decline",
        "stage_raw_score",
        "project_count",
        "data_source_building",
        "data_source_vacant",
        "data_source_landscape",
        "data_source_redevelopment",
    ]
    result = result[[c for c in output_cols if c in result.columns]].round(4)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("저장: %s (%d동) | 출처: %s", output_path, len(result), sources)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="서대문구 정비 사각지대 전처리 파이프라인")
    parser.add_argument("--mock", action="store_true", help="샘플(Mock) 데이터만 사용")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="API 실패 시 Mock 대체 없이 종료",
    )
    parser.add_argument("-o", "--output", default=OUTPUT_CSV, help="출력 CSV 경로")
    args = parser.parse_args()
    df = run_pipeline(
        use_mock=args.mock,
        allow_mock_fallback=not args.strict,
        output_path=args.output,
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
