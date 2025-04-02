#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于原始 TLE 选星代码的重构实现，并加入了信噪比（SNR）的融合：
  1. Satellite 类增加 snr 属性；
  2. update_satellite_info() 除了更新位置信息、仰角、方位角与 GEV 参数外，模拟赋予信噪比；
  3. OSSA 选星中，构造综合目标函数： composite_obj = w_geo * DGDOP + w_snr * (1 - normalized_avg_SNR)
  4. FCSDp 选星中，在簇内候选及补充过程中，利用综合指标：
         composite_weight = w_geo * (|beta - center| + |phi - reference|) + w_snr * (1 - normalized_SNR)
  5. MaxEle 算法保持原有，根据仰角选取，但结果中也保留了 SNR 信息供参考。
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import time
from datetime import datetime, timezone, timedelta
import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from skyfield.api import load, wgs84, EarthSatellite
import matplotlib.pyplot as plt
import random

# 为限定多线程数，设置环境变量



# ------------------------------
# 定义卫星数据结构
# ------------------------------
class Satellite:
    def __init__(self, sat_object):
        """
        参数:
          sat_object: Skyfield 库解析 TLE 后返回的 EarthSatellite 对象
        """
        self.sat_object = sat_object
        self.name = sat_object.name
        self.position = None      # ECEF 坐标（km）
        self.elevation = None     # 仰角（度）
        self.azimuth = None       # 方位角（度）
        self.phi = None           # 垂直特征角（弧度）
        self.beta = None          # 水平特征角（弧度）
        self.weight = None        # 综合评分（用于 FCSDp 算法补充和筛选）
        self.snr = None           # 信噪比（dB），实际中可采用真实测量值，此处进行模拟

    def __repr__(self):
        return f"Satellite({self.name}, SNR={self.snr:.2f} dB)"


# ------------------------------
# 更新卫星信息：位置、仰角、方位角、GEV 参数及 SNR（此处 SNR 为模拟值）
# ------------------------------
def update_satellite_info(sat, observer, t):
    """
    根据观测者 observer（wgs84.topos 对象）和时间 t 更新：
      - 卫星的 ECEF 坐标（km），存入 sat.position
      - 卫星的仰角、方位角（单位：度）
      - 根据仰角赋予 GEV 垂直角 phi（仰角>=50°，视为上层，赋值 2π/3；否则赋值 π/3）
      - beta 直接为方位角转换为弧度
      - 模拟赋予信噪比（SNR）属性，范围在 [20, 40] dB
    """
    # 获取 ITRS 坐标（单位 km）
    pos = sat.sat_object.at(t).position.m
    sat.position = pos

    # 根据观测者获得卫星观测数据（仰角、方位角）
    difference = sat.sat_object - observer
    alt, az, _ = difference.at(t).altaz()
    sat.elevation = alt.degrees
    sat.azimuth = az.degrees

    # 根据仰角确定垂直角 phi（GEV 参数）
    if sat.elevation >= 50:
        sat.phi = 2 * np.pi / 3
    else:
        sat.phi = np.pi / 3

    # 水平特征角 beta 为 azimuth 转换为弧度
    sat.beta = np.radians(sat.azimuth)

    # 模拟信噪比赋值（实际中应由硬件测量获得）
    sat.snr = random.uniform(20.0, 40.0)


# ------------------------------
# 计算组合的 DGDOP（基于伪观测矩阵）
# ------------------------------
def compute_dgdop(combo, user_pos):
    """
    对于给定卫星组合 combo（列表），构造伪观测矩阵 G，每行格式为：
         [e_x, e_y, e_z, 1]
    其中 e = (sat.position - user_pos) / ||sat.position - user_pos||。
    返回 sqrt(trace((G^T G)^{-1})) 作为 DGDOP 指标。
    """
    G = []
    for sat in combo:
        diff = sat.position - user_pos
        norm_diff = np.linalg.norm(diff)
        if norm_diff == 0:
            continue
        unit_vector = diff / norm_diff
        row = np.hstack((unit_vector, [1]))
        G.append(row)
    G = np.array(G)
    epsilon = 1e-8  # 增加微量项保证数值稳定性
    try:
        Q = np.linalg.inv(G.T @ G + epsilon * np.eye(G.shape[1]))
        dop = np.sqrt(np.trace(Q))
    except np.linalg.LinAlgError:
        dop = np.inf
    return dop


# ------------------------------
# OSSA 算法（最优全遍历法）——同时综合 DGDOP 与组合内卫星的 SNR
# ------------------------------
def ossa_selection(satellites, user_pos, n, w_geo=0.5, w_snr=0.5):
    """
    遍历所有可能的卫星组合（选取 n 颗），分别计算：
      - DGDOP：利用几何信息计算
      - 平均 SNR：计算组合内卫星的平均信噪比，并归一化（1 – normalized_avg_snr 越小越好）
    综合目标函数定义为：
         composite_obj = w_geo * DGDOP + w_snr * (1 - normalized_avg_SNR)
    返回使综合目标函数最小的组合作为最佳组合，以及目标函数值。
    """
    if not satellites:
        return None, np.inf

    # 计算全局 SNR 的最小和最大值
    snr_values = [sat.snr for sat in satellites]
    snr_min = min(snr_values)
    snr_max = max(snr_values)

    best_combo = None
    best_obj = np.inf
    for combo in combinations(satellites, n):
        dop_val = compute_dgdop(combo, user_pos)
        avg_snr = np.mean([sat.snr for sat in combo])
        normalized_avg_snr = (avg_snr - snr_min) / (snr_max - snr_min) if snr_max != snr_min else 1
        composite_obj = w_geo * dop_val + w_snr * (1 - normalized_avg_snr)
        if composite_obj < best_obj:
            best_obj = composite_obj
            best_combo = combo
    return best_combo, best_obj


# ------------------------------
# MaxEle 算法（最大仰角法）
# ------------------------------
def maxele_selection(satellites, user_pos, n):
    """
    将 360° 天区均分为 n 个区间，在每个区间中选取仰角最大的卫星。
    若选出卫星不足，则补充全局仰角最高的卫星。返回组合及对应 DGDOP。
    """
    bins = np.linspace(0, 360, n + 1)
    selected = []
    for i in range(n):
        sats_in_bin = [sat for sat in satellites if bins[i] <= sat.azimuth < bins[i+1]]
        if sats_in_bin:
            best_sat = max(sats_in_bin, key=lambda s: s.elevation)
            selected.append(best_sat)
    if len(selected) < n:
        remaining = [sat for sat in satellites if sat not in selected]
        remaining_sorted = sorted(remaining, key=lambda s: s.elevation, reverse=True)
        for sat in remaining_sorted:
            if len(selected) < n:
                selected.append(sat)
            else:
                break
    dop = compute_dgdop(selected, user_pos)
    return selected, dop


# ------------------------------
# FCSDp 算法（基于分层聚类与加权匹配）——在加权匹配中融合 SNR 综合评价
# ------------------------------
def fcsdp_selection(satellites, user_pos, n, w_geo=0.5, w_snr=0.5):
    """
    FCSDp 算法流程：
      1. 根据 GEV 垂直角 phi 将卫星分为上层（phi > π/2）和下层（phi <= π/2），
         其中上层参考值为 2π/3，下层参考值为 π/3。
      2. 分别对上层和下层卫星，根据水平特征角 beta 利用 KMeans 聚类，
         聚类数上层和下层分别为：
             - n 为偶数：各组均为 n/2；
             - n 为奇数：上层为 (n+1)//2，下层为 (n-1)//2
      3. 在每个聚类中选取候选卫星，其综合评价指标为：
             composite_weight = w_geo * (|beta - cluster_center| + |phi - reference|)
                                + w_snr * (1 - normalized_SNR)
         其中 normalized_SNR 为在所有候选卫星内归一化后的信噪比。
      4. 合并候选卫星；若候选不足 n，则从剩余卫星中按综合指标补充，
         若候选超出 n，则取综合指标最小的 n 颗。
      5. 返回最终选取组合及对应 DGDOP。
    """
    if not satellites:
        return [], np.inf

    # 计算全局 SNR 的最小和最大值
    snr_values = [sat.snr for sat in satellites]
    snr_min = min(snr_values)
    snr_max = max(snr_values)

    # 分组：上层与下层
    upper = [sat for sat in satellites if sat.phi > np.pi / 2]
    lower = [sat for sat in satellites if sat.phi <= np.pi / 2]

    upper_ref = 2 * np.pi / 3
    lower_ref = np.pi / 3

    # 根据 n 分配上层与下层所需候选数量
    if n % 2 == 0:
        k_upper = n // 2
        k_lower = n // 2
    else:
        k_upper = (n + 1) // 2
        k_lower = (n - 1) // 2

    # 对上层卫星利用 KMeans 聚类（基于 beta）
    selected_upper = []
    if upper and k_upper > 0:
        X_upper = np.array([[sat.beta] for sat in upper])
        kmeans_upper = KMeans(n_clusters=k_upper, random_state=42).fit(X_upper)
        centers_upper = kmeans_upper.cluster_centers_
        for cluster in range(k_upper):
            indices = np.where(kmeans_upper.labels_ == cluster)[0]
            best_sat = None
            best_weight = np.inf
            for i in indices:
                sat = upper[i]
                geo_component = abs(sat.beta - centers_upper[cluster][0]) + abs(sat.phi - upper_ref)
                normalized_snr = (sat.snr - snr_min) / (snr_max - snr_min) if snr_max != snr_min else 1
                snr_component = 1 - normalized_snr
                composite_weight = w_geo * geo_component + w_snr * snr_component
                if composite_weight < best_weight:
                    best_weight = composite_weight
                    best_sat = sat
            if best_sat is not None:
                selected_upper.append(best_sat)

    # 对下层卫星利用 KMeans 聚类（基于 beta）
    selected_lower = []
    if lower and k_lower > 0:
        X_lower = np.array([[sat.beta] for sat in lower])
        kmeans_lower = KMeans(n_clusters=k_lower, random_state=42).fit(X_lower)
        centers_lower = kmeans_lower.cluster_centers_
        for cluster in range(k_lower):
            indices = np.where(kmeans_lower.labels_ == cluster)[0]
            best_sat = None
            best_weight = np.inf
            for i in indices:
                sat = lower[i]
                geo_component = abs(sat.beta - centers_lower[cluster][0]) + abs(sat.phi - lower_ref)
                normalized_snr = (sat.snr - snr_min) / (snr_max - snr_min) if snr_max != snr_min else 1
                snr_component = 1 - normalized_snr
                composite_weight = w_geo * geo_component + w_snr * snr_component
                if composite_weight < best_weight:
                    best_weight = composite_weight
                    best_sat = sat
            if best_sat is not None:
                selected_lower.append(best_sat)

    # 合并上层和下层候选卫星
    selected = selected_upper + selected_lower

    # 若候选不足 n，则从剩余卫星中补充（按综合指标排序补充）
    if len(selected) < n:
        remaining = [sat for sat in satellites if sat not in selected]
        for sat in remaining:
            if sat.phi > np.pi / 2:
                geo_component = abs(sat.phi - upper_ref)
            else:
                geo_component = abs(sat.phi - lower_ref)
            normalized_snr = (sat.snr - snr_min) / (snr_max - snr_min) if snr_max != snr_min else 1
            snr_component = 1 - normalized_snr
            sat.weight = w_geo * geo_component + w_snr * snr_component
        remaining = sorted(remaining, key=lambda s: s.weight)
        while len(selected) < n and remaining:
            selected.append(remaining.pop(0))
    # 若候选超过 n，则取综合指标最小的 n 颗
    elif len(selected) > n:
        for sat in selected:
            if sat.phi > np.pi / 2:
                geo_component = abs(sat.phi - upper_ref)
            else:
                geo_component = abs(sat.phi - lower_ref)
            normalized_snr = (sat.snr - snr_min) / (snr_max - snr_min) if snr_max != snr_min else 1
            snr_component = 1 - normalized_snr
            sat.weight = w_geo * geo_component + w_snr * snr_component
        selected = sorted(selected, key=lambda s: s.weight)[:n]

    dop = compute_dgdop(selected, user_pos)
    return selected, dop


# ------------------------------
# 判断卫星是否可视（仰角大于一定阈值，这里取 15°）
# ------------------------------
def is_visible(sat, observer, t, alt_threshold=15):
    difference = sat.sat_object - observer
    alt, _, _ = difference.at(t).altaz()
    return alt.degrees > alt_threshold


# ------------------------------
# 主程序：加载 TLE 数据，更新各卫星信息，并调用各种选星算法
# ------------------------------
def main():
    # TLE 文件路径（请根据实际情况修改）
    tle_file = 'satellite_data/starlink_tle_20250106_144506.txt'
    satellites_tle = load.tle_file(tle_file)
    print(f"加载了 {len(satellites_tle)} 颗卫星的 TLE 数据.")

    # 将 Skyfield 的 EarthSatellite 对象转换为自定义 Satellite 对象
    satellites = [Satellite(sat) for sat in satellites_tle]

    # 设置观测时间（例如 UTC 2025-01-06 16:00:00）
    ts = load.timescale()
    t = ts.from_datetime(datetime(2025, 1, 6, 16, 0, 0, tzinfo=timezone.utc))

    # 定义观测者位置（例如：[45.75°N, 126.68°E, 100 m]）
    observer = wgs84.latlon(45.75, 126.68, elevation_m=100)

    # 过滤出仰角大于 15° 的可视卫星
    satellites = [sat for sat in satellites if is_visible(sat, observer, t)]

    # 将用户位置转换为 ITRS 坐标（km），用于 DGDOP 计算
    user_pos = observer.at(t).position.m

    # 更新每颗卫星的信息（位置、仰角、方位角、GEV 参数、以及模拟SNR）
    for sat in satellites:
        update_satellite_info(sat, observer, t)

    # 设置选星卫星数量（例如选择 4 颗）
    n = 4
    print(f"\n选取 {n} 颗卫星:")

    # 参数设置：w_geo 与 w_snr 分别为几何与信噪比在综合目标函数中的权重
    w_geo = 0.5
    w_snr = 0.5

    # OSSA 算法（综合 DGDOP 与 SNR）
    start = time.perf_counter()
    combo_ossa, obj_ossa = ossa_selection(satellites, user_pos, n, w_geo, w_snr)
    elapsed_ossa = (time.perf_counter() - start) * 1000
    # 同时计算组合的 DGDOP（仅几何指标）
    dop_ossa = compute_dgdop(combo_ossa, user_pos)
    print("\n[OSSA 算法]")
    print("选中的卫星组合:", combo_ossa)
    print(f"DGDOP = {dop_ossa:.2f}, 综合目标 = {obj_ossa:.4f}, 耗时 = {elapsed_ossa:.4f} ms")

    # MaxEle 算法（基于最大仰角，未融入 SNR 综合评价）
    start = time.perf_counter()
    combo_maxele, dop_maxele = maxele_selection(satellites, user_pos, n)
    elapsed_maxele = (time.perf_counter() - start) * 1000
    print("\n[MaxEle 算法]")
    print("选中的卫星组合:", combo_maxele)
    print(f"DGDOP = {dop_maxele:.2f}, 耗时 = {elapsed_maxele:.4f} ms")

    # FCSDp 算法（在聚类匹配中加入了 SNR 综合评价）
    start = time.perf_counter()
    combo_fcsdp, dop_fcsdp = fcsdp_selection(satellites, user_pos, n, w_geo, w_snr)
    elapsed_fcsdp = (time.perf_counter() - start) * 1000
    print("\n[FCSDp 算法]")
    print("选中的卫星组合:", combo_fcsdp)
    print(f"DGDOP = {dop_fcsdp:.2f}, 耗时 = {elapsed_fcsdp:.4f} ms")

    # ------------------------------
    # 可视化对比（可选）
    # ------------------------------
    # x = np.arange(3)
    # dop_values = [dop_ossa, dop_fcsdp, dop_maxele]
    # times = [elapsed_ossa, elapsed_fcsdp, elapsed_maxele]
    # labels = ['OSSA', 'FCSDp', 'MaxEle']
    #
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # axs[0].bar(x, dop_values, tick_label=labels)
    # axs[0].set_ylabel('DGDOP')
    # axs[0].set_title('DGDOP 对比')
    #
    # axs[1].bar(x, times, tick_label=labels)
    # axs[1].set_ylabel('耗时 (ms)')
    # axs[1].set_title('执行时间对比')
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()