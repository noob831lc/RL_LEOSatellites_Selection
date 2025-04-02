import time
from datetime import datetime, timezone, timedelta
import os
import math
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from skyfield.api import load, wgs84, EarthSatellite
from skyfield.framelib import itrs, tirs
import matplotlib.pyplot as plt


# ------------------------------
# 定义卫星数据结构
# ------------------------------
class Satellite:
    def __init__(self, sat_object):
        """
        sat_object: Skyfield 库中解析 TLE 后返回的 EarthSatellite 对象
        """
        self.sat_object = sat_object
        self.name = sat_object.name
        # 用于 DGDOP 计算：卫星的 ECEF 坐标（km）
        self.position = None
        # 用于天顶角、方位角计算（由观测者计算得到，单位：度）
        self.elevation = None
        self.azimuth = None
        # GEV 特征角（用于 FCSDp 中分层聚类分析），单位：弧度
        # 这里简单用仰角来区分：若卫星高度较高则赋值 2π/3，否则赋值 π/3
        self.phi = None
        # 水平特征角：直接取方位角转为弧度
        self.beta = None
        # 权重，在 FCSDp 算法中用于排序（综合两个角度的距离）
        self.weight = None

    def __repr__(self):
        return f"{self.name}"


# ------------------------------
# Skyfield 辅助函数：更新卫星信息
# ------------------------------
def update_satellite_info(sat, observer, t):
    """
    根据观测者位置 observer（wgs84.topos 对象）和时间 t 更新卫星信息：
      - 计算卫星的 ECEF 坐标（ITRS），存入 sat.position（km）
      - 利用观测者获得卫星的仰角、方位角（通过 Skyfield 的 altaz() 方法）
      - 根据仰角设置 GEV 的垂直角 phi；并将方位角转为弧度作为 beta
    """
    # 利用 satellite 对象获得 ITRS 坐标（单位 km）
    pos = sat.sat_object.at(t).position
    sat.position = pos.m  # 3D 坐标
    # print(sat.position)
    difference = sat.sat_object - observer

    # 使用观测者对象获得卫星的观测数据（alt, az, distance）
    alt, az, distance = difference.at(t).altaz()
    sat.elevation = alt.degrees  # 仰角，单位：度

    sat.azimuth = az.degrees  # 方位角，单位：度

    # 根据仰角设置 GEV 的垂直角 phi，
    # 这里简单约定：若仰角>=50°认为较优（上层），赋值 2π/3；否则赋值 π/3。
    if sat.elevation >= 50:
        sat.phi = 2 * np.pi / 3
    else:
        sat.phi = np.pi / 3

    # beta 直接取 azimuth 转换为弧度
    sat.beta = np.radians(sat.azimuth)


# ------------------------------
# 计算组合的 DGDOP（以伪观测矩阵的指标作为评价）
# ------------------------------
def compute_dgdop(combo, user_pos):
    """
    对于给定卫星组合 combo（列表），构造观测矩阵 G，每行格式：
         [e_x, e_y, e_z, 1]
    其中 e = (sat.position - user_pos) / ||sat.position - user_pos||。
    然后计算 Q = (G^T G)^{-1}，以 sqrt(trace(Q)) 作为 DGDOP 指标。
    user_pos：观测者在 ITRS 中的坐标（km）
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

    try:
        epsilon = 1e-8
        Q = np.linalg.inv(G.T @ G + epsilon * np.eye(G.shape[1]))
        # Q = np.linalg.inv(G.T @ G )
        dop = np.sqrt(abs(np.trace(Q)))
    except np.linalg.LinAlgError:
        dop = np.inf
    return 10*dop


# ------------------------------
# OSSA 算法实现（最优全遍历法）
# ------------------------------
def ossa_selection(satellites, user_pos, n):
    """
    OSSA 算法：
      遍历所有可能的卫星组合（n 颗），计算每个组合的 DGDOP，
      并返回 DGDOP 最小的组合作为最佳组合。
    数学上，即求解：
          C_opt = argmin_{C⊆Ω, |C|=n} √(trace((G_C^T G_C)^{-1}))
    """
    best_combo = None
    best_dop = np.inf
    for combo in combinations(satellites, n):
        dop_val = compute_dgdop(combo, user_pos)
        if dop_val < best_dop:
            best_dop = dop_val
            best_combo = combo
    return best_combo, best_dop


# ------------------------------
# MaxEle 算法实现（最大仰角法）
# ------------------------------
def maxele_selection(satellites, user_pos, n):
    """
    MaxEle 算法：
      将 360° 天区按照 n 个区间均等划分，然后在每个区间内选取仰角最大的卫星。
      数学上，对每个区间 R_j, 选出：
             S*_j = argmax_{S_i ∈ R_j} θ_i
      最终组合：
             C_MaxEle = {S*_1, S*_2, …, S*_n}
    """
    # 先确保每颗卫星都有 elevation、azimuth 信息（一般在 update_satellite_info 中已更新）
    bins = np.linspace(0, 360, n + 1)  # 划分 n 个区间
    selected = []
    for i in range(n):
        sats_in_bin = [sat for sat in satellites if bins[i] <= sat.azimuth < bins[i + 1]]
        if sats_in_bin:
            # 在该区间内选择仰角最大的卫星
            best_sat = max(sats_in_bin, key=lambda sat: sat.elevation)
            selected.append(best_sat)
    # 如果选出卫星不足 n 个，则补充全局仰角最高的卫星
    if len(selected) < n:
        remaining = [sat for sat in satellites if sat not in selected]
        remaining = sorted(remaining, key=lambda s: s.elevation, reverse=True)
        for sat in remaining:
            selected.append(sat)
            if len(selected) == n:
                break
    dop = compute_dgdop(selected, user_pos)
    return selected, dop


# ------------------------------
# FCSDp 算法实现（基于分层聚类和加权匹配的方法）
# ------------------------------
def fcsdp_selection(satellites, user_pos, n):
    """
    FCSDp 算法主要步骤：
      1. 对每颗卫星利用已更新的信息（elevation, azimuth）赋予 GEV 特征角：phi 和 beta；
      2. 将卫星根据 phi 分为两组：上层（phi > π/2）与下层（phi <= π/2），同时设定参考 label-1：
            上层参考：2π/3； 下层参考：π/3；
      3. 针对每组分别按照 beta 利用 K-means++ 聚类，聚类数量根据 n 分配：
            当 n 为偶数：上、下组各 n/2；若 n 为奇数：上组 (n+1)//2，下组 (n-1)//2；
      4. 在每个聚类中，选取距离 (|beta - cluster_center| + |phi - label|) 最小的卫星作为候选；
      5. 合并候选结果：如果候选不足则从剩余卫星中补充，如果过多则按权重排序选取前 n 个；
      6. 返回最终组合及对应的 DGDOP。
    """
    # 分组：上层与下层
    upper = [sat for sat in satellites if sat.phi > np.pi / 2]
    lower = [sat for sat in satellites if sat.phi <= np.pi / 2]

    # 设定参考 label-1
    upper_label1 = 2 * np.pi / 3
    lower_label1 = np.pi / 3

    # 确定上层与下层的聚类数目
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

    # 合并上层与下层的候选卫星
    selected = selected_upper + selected_lower

    # 若候选不足 n 个，则从剩余未选卫星中补充，方法：根据 |phi - label|（对应组别）排序
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
    # 若候选过多，则选择权重最小的 n 个
    elif len(selected) > n:
        for sat in selected:
            if sat.phi > np.pi / 2:
                sat.weight = abs(sat.phi - upper_label1)
            else:
                sat.weight = abs(sat.phi - lower_label1)
        selected = sorted(selected, key=lambda s: s.weight)[:n]

    dop = compute_dgdop(selected, user_pos)
    return selected, dop


def visable(sat, observer, t):
    difference = sat.sat_object - observer
    topocentric = difference.at(t)
    alt, az, distance = topocentric.altaz()
    if alt.degrees > 25:
        return True
    return False


def visable_p(sat, observer, start_time_utc):
    difference = sat.sat_object - observer
    time_list = [ts.from_datetime(start_time_utc + timedelta(seconds=s)) for s in range(120)]
    for t in time_list:
        topocentric = difference.at(t)
        alt, az, distance = topocentric.altaz()
        if alt.degrees < 25:
            return False
    return True


# ------------------------------
# 主程序：加载 TLE 文件，更新卫星信息，并调用各算法
# ------------------------------
if __name__ == '__main__':
    # 加载 TLE（本地文件），文件格式应符合常见 TLE 格式，每 3 行代表一颗卫星
    tle_file = 'satellite_data\\starlink_tle_20250106_144506.txt'  # 替换为你本地 TLE 文件的路径
    satellites_tle = load.tle_file(tle_file)
    print(f"加载了 {len(satellites_tle)} 颗卫星 TLE 数据.")

    # 将 Skyfield 的 EarthSatellite 对象转换为我们自定义的 Satellite 对象列表
    satellites = [Satellite(sat) for sat in satellites_tle]

    # 定义时间（使用当前时刻）
    ts = load.timescale()
    t = ts.from_datetime(datetime(2025, 1, 6, 14, 0, 0, tzinfo=timezone.utc))

    # 定义观测者（用户）位置（可采用文中示例：[126.68°E, 45.75°N, 100 m]）
    observer = wgs84.latlon(45.75, 126.68, elevation_m=100)
    satellites = [sat for sat in satellites if visable(sat, observer, t)]

    # 转换用户位置到 ITRS 坐标（km），用于 DGDOP 计算
    user_itrs = observer.at(t).position.m

    # 更新每颗卫星相对于观测者的位置、仰角、方位角、GEV 参数
    for sat in satellites:
        update_satellite_info(sat, observer, t)

    # 设置所需选择卫星的数量（例如选择 4 颗）

    n_sat_counts = [4]
    # 定义三种算法名称（对应 OSSA、MaxEle、FCSDp）
    algorithms = ['OSSA', 'FCSDp', 'MaxEle']
    # 用字典保存不同算法在不同 n 下得到的 DGDOP 和执行时间
    results_dgdop = {alg: [] for alg in algorithms}
    results_time = {alg: [] for alg in algorithms}
    for n in n_sat_counts:
        print(f"\n测试卫星数量 n = {n}")
        # OSSA 算法
        start = time.perf_counter()
        combo, dop = ossa_selection(satellites, user_itrs, n)
        elapsed = (time.perf_counter() - start) * 1000
        results_dgdop['OSSA'].append(dop)
        results_time['OSSA'].append(elapsed)
        print("OSSA选择的卫星组合:", combo)
        print(f"OSSA: DGDOP = {dop:.2f}, Time = {elapsed:.4f} ms")

        # MaxEle 算法
        start = time.perf_counter()
        combo, dop = maxele_selection(satellites, user_itrs, n)
        elapsed = (time.perf_counter() - start) * 1000
        print("MaxEle选择的卫星组合:", combo)
        results_dgdop['MaxEle'].append(dop)
        results_time['MaxEle'].append(elapsed)
        print(f"MaxEle: DGDOP = {dop:.2f}, Time = {elapsed:.4f} ms")

        # FCSDp 算法
        start = time.perf_counter()
        combo, dop = fcsdp_selection(satellites, user_itrs, n)
        elapsed = (time.perf_counter() - start) * 1000
        print("FCSDp选择的卫星组合:", combo)
        results_dgdop['FCSDp'].append(dop)
        results_time['FCSDp'].append(elapsed)
        print(f"FCSDp: DGDOP = {dop:.2f}, Time = {elapsed:.4f} ms")


        

    # ------------------------------
    # 利用 matplotlib 绘制对比结果
    # ------------------------------
    # 设置柱状图位置、宽度
    width = 0.25
    x = np.arange(len(n_sat_counts))  # x 轴刻度位置

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[1].set_yscale('log')
    # 绘制 DGDOP 对比图
    axes[0].bar(x - width, results_dgdop['OSSA'], width, label='OSSA')
    axes[0].bar(x, results_dgdop['FCSDp'], width, label='FCSDp')
    axes[0].bar(x + width, results_dgdop['MaxEle'], width, label='MaxEle')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(n_sat_counts)
    axes[0].set_xlabel('sat num')
    axes[0].set_ylabel('DGDOP')
    axes[0].set_title('DGDOP ')
    axes[0].legend()

    # 绘制 执行时间 对比图
    axes[1].bar(x - width, results_time['OSSA'], width, label='OSSA')
    axes[1].bar(x, results_dgdop['FCSDp'], width, label='FCSDp')
    axes[1].bar(x + width, results_time['MaxEle'], width, label='MaxEle')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(n_sat_counts)
    axes[1].set_xlabel('num')
    axes[1].set_ylabel('time (ms)')
    axes[1].set_title('time')
    # axes[1].legend()

    plt.tight_layout()
    plt.show()
