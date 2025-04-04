import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans

def compute_dgdop(combo, user_pos):
    """
    计算给定卫星组合 combo 的 DGDOP 指标(单位 m)
    combo: 卫星列表(Satellite 对象)
    user_pos: 观测者位置(单位 m)
    """
    G = []
    for sat in combo:
        diff = sat.position - user_pos
        norm_diff = np.linalg.norm(diff)
        if norm_diff == 0:
            continue
        unit_vector = diff / norm_diff
        row = np.hstack([unit_vector, [1]])
        G.append(row)
    G = np.array(G)
    epsilon = 1e-8
    try:
        Q = np.linalg.inv(G.T @ G + epsilon * np.eye(G.shape[1]))
        dop = np.sqrt(np.trace(Q))
    except np.linalg.LinAlgError:
        dop = np.inf
    return 10 * dop


def ossa_selection(satellites, user_pos, n):
    """
    OSSA 算法：遍历所有可能的 n 颗卫星组合，返回 DGDOP 最小的组合作为最佳方案
    """
    best_combo = None
    best_dop = np.inf
    for combo in combinations(satellites, n):
        dop_val = compute_dgdop(combo, user_pos)
        if dop_val < best_dop:
            best_dop = dop_val
            best_combo = combo
    return best_combo, best_dop


def maxele_selection(satellites, user_pos, n):
    """
    MaxEle 算法：将 360° 天区均分为 n 个区间，在每个区间内选取仰角最大的卫星
    """
    bins = np.linspace(0, 360, n + 1)
    selected = []
    for i in range(n):
        sats_in_bin = [sat for sat in satellites if bins[i] <= sat.azimuth < bins[i + 1]]
        if sats_in_bin:
            best_sat = max(sats_in_bin, key=lambda sat: sat.elevation)
            selected.append(best_sat)
    if len(selected) < n:
        remaining = [sat for sat in satellites if sat not in selected]
        remaining = sorted(remaining, key=lambda s: s.elevation, reverse=True)
        for sat in remaining:
            selected.append(sat)
            if len(selected) == n:
                break
    dop = compute_dgdop(selected, user_pos)
    return selected, dop

def fcsdp_selection(satellites, user_pos, n):
    """
    FCSDp 算法：基于分层聚类与加权匹配实现卫星选择。
    将卫星根据 phi 分为两组（上层与下层），各自利用 K-means 聚类后选出候选卫星，
    最后合并候选结果并调整至目标数量 n。
    """
    upper = [sat for sat in satellites if sat.phi > np.pi / 2]
    lower = [sat for sat in satellites if sat.phi <= np.pi / 2]

    upper_label1 = 2 * np.pi / 3
    lower_label1 = np.pi / 3

    if n % 2 == 0:
        k_upper = n // 2
        k_lower = n // 2
    else:
        k_upper = (n + 1) // 2
        k_lower = (n - 1) // 2

    selected_upper = []
    if upper and k_upper > 0:
        X_upper = np.array([[sat.beta] for sat in upper])
        kmeans_upper = KMeans(n_clusters=k_upper, random_state=0).fit(X_upper)
        centers_upper = kmeans_upper.cluster_centers_
        for clus_label in range(k_upper):
            indices = np.where(kmeans_upper.labels_ == clus_label)[0]
            best_sat = None
            best_weight = np.inf
            for i in indices:
                sat = upper[i]
                cur_weight = abs(sat.beta - centers_upper[clus_label][0]) + abs(sat.phi - upper_label1)
                if cur_weight < best_weight:
                    best_weight = cur_weight
                    best_sat = sat
            if best_sat is not None:
                selected_upper.append(best_sat)

    selected_lower = []
    if lower and k_lower > 0:
        X_lower = np.array([[sat.beta] for sat in lower])
        kmeans_lower = KMeans(n_clusters=k_lower, random_state=0).fit(X_lower)
        centers_lower = kmeans_lower.cluster_centers_
        for clus_label in range(k_lower):
            indices = np.where(kmeans_lower.labels_ == clus_label)[0]
            best_sat = None
            best_weight = np.inf
            for i in indices:
                sat = lower[i]
                cur_weight = abs(sat.beta - centers_lower[clus_label][0]) + abs(sat.phi - lower_label1)
                if cur_weight < best_weight:
                    best_weight = cur_weight
                    best_sat = sat
            if best_sat is not None:
                selected_lower.append(best_sat)

    selected = selected_upper + selected_lower
    if len(selected) < n:
        remaining = [sat for sat in satellites if sat not in selected]
        for sat in remaining:
            if sat.phi > np.pi / 2:
                sat.weight = abs(sat.phi - upper_label1)
            else:
                sat.weight = abs(sat.phi - lower_label1)
        remaining = sorted(remaining, key=lambda s: s.weight)
        while len(selected) < n and remaining:
            selected.append(remaining.pop(0))
    elif len(selected) > n:
        for sat in selected:
            if sat.phi > np.pi / 2:
                sat.weight = abs(sat.phi - upper_label1)
            else:
                sat.weight = abs(sat.phi - lower_label1)
        selected = sorted(selected, key=lambda s: s.weight)[:n]

    dop = compute_dgdop(selected, user_pos)
    return selected, dop