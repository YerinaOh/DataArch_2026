#!/usr/bin/env python3
"""
서대문구 정비 사각지대 — K-Means 군집 분석

입력: seodaemun_clean_data.csv
출력: seodaemun_result.csv (cluster, region_type 포함)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_CSV = "seodaemun_clean_data.csv"
OUTPUT_CSV = "seodaemun_result.csv"
N_CLUSTERS = 4
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "obsolescence_score",
    "decline_score",
    "development_score",
]

REGION_TYPES: list[str] = [
    "정비사각지대",
    "정비중 지역",
    "양호지역",
    "잠재쇠퇴지역",
]


def score_centroid_for_region_type(
    centroid: np.ndarray,
    region_type: str,
    mids: tuple[float, float, float],
) -> float:
    """
    중심점 (노후도, 쇠퇴도, 개발현황)이 각 유형 정의에 얼마나 부합하는지 점수화.
    점수가 높을수록 해당 유형에 적합.
    """
    obs, decl, dev = centroid
    o_mid, d_mid, v_mid = mids

    if region_type == "정비사각지대":
        # 노후도 고 / 쇠퇴도 고 / 개발도 저
        return float(obs + decl - dev)
    if region_type == "정비중 지역":
        # 노후도 고 / 쇠퇴도 고 / 개발도 고
        return float(obs + decl + dev)
    if region_type == "양호지역":
        # 노후도 저 / 쇠퇴도 저
        return float(-(obs + decl))
    if region_type == "잠재쇠퇴지역":
        # 노후도 중 / 쇠퇴도 중 / 개발도 중
        return float(-(abs(obs - o_mid) + abs(decl - d_mid) + abs(dev - v_mid)))
    raise ValueError(f"알 수 없는 유형: {region_type}")


def map_clusters_to_region_types(centroids: np.ndarray) -> dict[int, str]:
    """
    K-Means 클러스터 중심점을 4개 region_type에 일대일 매핑.
    헝가리안 알고리즘으로 전역 최적 배정.
    """
    if centroids.shape[0] != N_CLUSTERS:
        raise ValueError(f"클러스터 수는 {N_CLUSTERS}개여야 합니다.")

    mids = (
        float(np.median(centroids[:, 0])),
        float(np.median(centroids[:, 1])),
        float(np.median(centroids[:, 2])),
    )

    cost = np.zeros((N_CLUSTERS, N_CLUSTERS), dtype=float)
    for i in range(N_CLUSTERS):
        for j, region_type in enumerate(REGION_TYPES):
            cost[i, j] = -score_centroid_for_region_type(centroids[i], region_type, mids)

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {int(row): REGION_TYPES[int(col)] for row, col in zip(row_ind, col_ind)}

    logger.info("중심점 → 유형 매핑:")
    for cluster_id in sorted(mapping):
        c = centroids[cluster_id]
        logger.info(
            "  cluster %d → %s (노후=%.2f, 쇠퇴=%.2f, 개발=%.2f)",
            cluster_id,
            mapping[cluster_id],
            c[0],
            c[1],
            c[2],
        )
    return mapping


def run_cluster_analysis(
    input_path: str | Path = INPUT_CSV,
    output_path: str | Path = OUTPUT_CSV,
    n_clusters: int = N_CLUSTERS,
) -> pd.DataFrame:
    if n_clusters != len(REGION_TYPES):
        raise ValueError(
            f"region_type 자동 매핑은 군집 수 {len(REGION_TYPES)}개에서만 지원합니다."
        )

    df = pd.read_csv(input_path, encoding="utf-8-sig")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    X = df[FEATURE_COLUMNS].astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # 원 스케일 중심점 (표준화 역변환)
    centroids_scaled = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)

    cluster_to_type = map_clusters_to_region_types(centroids)

    result = df.copy()
    result["cluster"] = labels
    result["region_type"] = result["cluster"].map(cluster_to_type)

    # 중심점 참고용 컬럼 (동일 cluster 내 모든 행에 동일 값)
    for i in range(n_clusters):
        mask = result["cluster"] == i
        result.loc[mask, "centroid_obsolescence"] = centroids[i, 0]
        result.loc[mask, "centroid_decline"] = centroids[i, 1]
        result.loc[mask, "centroid_development"] = centroids[i, 2]

    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("저장 완료: %s (%d개 동)", output_path, len(result))

    summary = (
        result.groupby(["cluster", "region_type"], as_index=False)
        .agg(
            dong_count=("dong", "count"),
            obsolescence_mean=("obsolescence_score", "mean"),
            decline_mean=("decline_score", "mean"),
            development_mean=("development_score", "mean"),
        )
        .sort_values("cluster")
    )
    logger.info("군집 요약:\n%s", summary.to_string(index=False))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="서대문구 K-Means 군집 분석")
    parser.add_argument(
        "-i",
        "--input",
        default=INPUT_CSV,
        help=f"전처리 CSV (기본: {INPUT_CSV})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=OUTPUT_CSV,
        help=f"결과 CSV (기본: {OUTPUT_CSV})",
    )
    parser.add_argument(
        "-k",
        "--clusters",
        type=int,
        default=N_CLUSTERS,
        help=f"군집 수 (기본: {N_CLUSTERS})",
    )
    args = parser.parse_args()

    result = run_cluster_analysis(
        input_path=args.input,
        output_path=args.output,
        n_clusters=args.clusters,
    )
    print(result[["dong", "cluster", "region_type"] + FEATURE_COLUMNS].to_string(index=False))


if __name__ == "__main__":
    main()
