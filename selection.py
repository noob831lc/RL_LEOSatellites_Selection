import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


# ------------------------------
# 卫星数据结构定义
# ------------------------------
class Satellite:
    def __init__(self, sat_id, position):
        """
        sat_id: 卫星标识
        position: 卫星的三维坐标（假设为 ECEF 坐标），例如 np.array([x, y, z])
        """
        self.sat_id = sat_id
        self.position = np.array(position, dtype=float)
        # 以下属性会在后面计算，单位均为角度（elevation, azimuth）或者弧度（phi, beta）
        self.elevation = None  # 仰角，单位：度
        self.azimuth = None  # 方位角，单位：度
        self.phi = None  # GEV 的垂直特征角（弧度）；本例中简单处理：高仰角卫星赋值较大
        self.beta = None  # GEV 的水平方向特征角（弧度）
        self.weight = None  # 在 FCSDp 中计算得到的权重

    def __repr__(self):
        return f"Sat({self.sat_id})"


# ------------------------------
# 辅助函数
# ------------------------------
def compute_elevation_azimuth(sat_pos, user_pos):
    """
    根据用户位置 user_pos 和卫星位置 sat_pos 计算仰角和方位角。
    为简化问题，这里假设用户局部坐标系中 z 轴垂直向上，
    仰角：diff[2] / ||diff||
    方位角：atan2(diff_y, diff_x) （结果转换为 0°-360°区间的度数）
    """
    diff = sat_pos - user_pos
    norm = np.linalg.norm(diff)
    if norm == 0:
        return 0.0, 0.0
    elevation = np.degrees(np.arcsin(diff[2] / norm))
    azimuth = np.degrees(np.arctan2(diff[1], diff[0]))
    if azimuth < 0:
        azimuth += 360
    return elevation, azimuth


def compute_dgdop(combo, user_pos):
    """
    根据组合中各卫星的伪观测矩阵计算 DGDOP 值。
    模型：
      对于每颗卫星，假设伪观测方程可写为：
      ρ = ||X_sat - x_use|| + tb, 线性化后，
      每行为 [e_x, e_y, e_z, 1]，其中 e=(X_sat - x_use)/||X_sat - x_use||
      则构成矩阵 G，定义 DOP = sqrt(tr((G^T G)^{-1}))
    """
    G = []
    for sat in combo:
        diff = sat.position - user_pos
        norm = np.linalg.norm(diff)
        if norm == 0:
            continue
        unit_vector = diff / norm
        # 这里构造的每一行包含 3 个方向余弦和常数项
        row = np.hstack([unit_vector, [1]])
        G.append(row)
    G = np.array(G)
    try:
        Q = np.linalg.inv(G.T @ G)
        dop = np.sqrt(np.trace(Q))
    except np.linalg.LinAlgError:
        dop = np.inf
    return dop


# ------------------------------
# 1. OSSA 算法实现
# ------------------------------
def ossa_selection(satellites, user_pos, n):
    """
    OSSA 算法：遍历所有可能的卫星组合，求出使 DGDOP 最小的组合
    数学上：求解
      C_opt = arg min_{C ⊆ Ω, |C| = n} sqrt(tr((G_C^T G_C)^{-1}))
    """
    best_combo = None
    best_dop = np.inf
    # 采用 itertools.combinations 计算所有 n 组合（当总卫星数 m 较小时可行）
    for combo in combinations(satellites, n):
        dop_val = compute_dgdop(combo, user_pos)
        if dop_val < best_dop:
            best_dop = dop_val
            best_combo = combo
    return best_combo, best_dop


# ------------------------------
# 2. MaxEle 算法实现
# ------------------------------
def maxele_selection(satellites, user_pos, n):
    """
    MaxEle 算法：将天空按照方位角进行均等划分，每个区域内选择仰角最大的卫星。
    数学上：
      对于区域 R_j, 选出
          S*_j = arg max_{S_i ∈ R_j} θ_i,
      最终组合为
          C_MaxEle = {S*_1, S*_2, ..., S*_n}
    """
    # 首先确保每颗卫星已有 elevation 和 azimuth 信息
    for sat in satellites:
        if sat.elevation is None or sat.azimuth is None:
            sat.elevation, sat.azimuth = compute_elevation_azimuth(sat.position, user_pos)

    # 将 [0, 360) 划分为 n 个区间
    bins = np.linspace(0, 360, n + 1)
    selected = []
    for i in range(n):
        # 选择 azimuth 落在该区间内的卫星
        sat_in_bin = [sat for sat in satellites if bins[i] <= sat.azimuth < bins[i + 1]]
        if sat_in_bin:
            # 在该区间中选取仰角最大的卫星
            best_sat = max(sat_in_bin, key=lambda sat: sat.elevation)
            selected.append(best_sat)
    # 若选出的卫星不足 n 个，则补充全局最高仰角的卫星
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
# 3. FCSDp 算法实现
# ------------------------------
def fcsdp_selection(satellites, user_pos, n):
    """
    FCSDp 算法：
    1. 对每颗卫星计算其 GEV 特征，取两个角度 φ 与 β；
       此处我们利用仰角和方位角的简单函数关系模拟：
         - 假设当仰角 >= 50度时，认为卫星具有较优的几何特性，赋值 φ = 2π/3  否则 φ = π/3
         - β 直接取 azimuth 转换为弧度
    2. 将卫星分为两组：上层（φ > π/2）和下层（φ <= π/2）。对上层设定 label-1 为 2π/3，对下层设定 label-1 为 π/3；
    3. 分别对两组卫星按照 β 利用 K-means++ 进行聚类（上层和下层聚类数量取决于所需卫星数）；
    4. 在每个聚类中选择距离（|β - 聚类中心| + |φ - label1|）最小的卫星作为候选；
    5. 合并候选卫星构成组合，如果数量不足则从剩余卫星中补充；如果过多则排序后取前 n 个；
    6. 对选出组合计算 DGDOP。
    """
    # 步骤1：确保每颗卫星具有 elevation、azimuth，如果没有，则计算之
    for sat in satellites:
        if sat.elevation is None or sat.azimuth is None:
            sat.elevation, sat.azimuth = compute_elevation_azimuth(sat.position, user_pos)
        # 模拟 GEV 的 φ：此处简单根据 elevation 赋予两个可能的数值
        if sat.elevation >= 50:
            sat.phi = 2 * np.pi / 3  # 上层
        else:
            sat.phi = np.pi / 3  # 下层
        sat.beta = np.radians(sat.azimuth)  # 采用 azimuth 作 β，弧度制

    # 步骤2：分组：上层与下层
    upper = [sat for sat in satellites if sat.phi > np.pi / 2]
    lower = [sat for sat in satellites if sat.phi <= np.pi / 2]

    # 步骤3：设定 label-1
    upper_label1 = 2 * np.pi / 3
    lower_label1 = np.pi / 3

    # 步骤4：根据待选卫星数 n 分别设置聚类数量：若 n 为偶数，则两个分组均为 n/2；若 n 为奇数则上层取 (n+1)//2，下层取 (n-1)//2
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
        # 在每个聚类中选择距离 (|β - cluster_center| + |φ - upper_label1|) 最小的卫星
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

    # 步骤5：合并两组候选卫星
    selected = selected_upper + selected_lower

    # 如果候选不足，则从剩余卫星中补充，按照简单的 |φ - label1| 距离排序选取
    if len(selected) < n:
        remaining = [sat for sat in satellites if sat not in selected]
        # 对于 remaining，根据所属组选择对应的 label-1
        for sat in remaining:
            if sat.phi > np.pi / 2:
                sat.weight = abs(sat.phi - upper_label1)
            else:
                sat.weight = abs(sat.phi - lower_label1)
        remaining = sorted(remaining, key=lambda s: s.weight)
        while len(selected) < n and remaining:
            selected.append(remaining.pop(0))
    # 如果候选过多则选择权重最小的 n 个
    elif len(selected) > n:
        for sat in selected:
            if sat.phi > np.pi / 2:
                sat.weight = abs(sat.phi - upper_label1)
            else:
                sat.weight = abs(sat.phi - lower_label1)
        selected = sorted(selected, key=lambda s: s.weight)[:n]

    # 【扩展】如果需要考虑多历元中时间匹配，可构造权重矩阵后采用匈牙利算法，
    # 这里为简化处理，我们直接返回选出的组合。
    dop = compute_dgdop(selected, user_pos)
    return selected, dop


# ------------------------------
# 示例与测试
# ------------------------------
def generate_synthetic_satellites(num):
    """
    生成 num 颗简单模拟的卫星，每颗卫星的位置在一定范围内随机分布。
    这里假设用户位置为原点，卫星距离在 20200 km 附近（GPS 典型高度）或其它较低值（LEO高度可调整）。
    本例中用于描述算法选择原理，数值选取仅供说明。
    """
    satellites = []
    for i in range(num):
        # 随机生成 3D 坐标（这里采用范围 [-15000, 15000] 公里内的随机点，并保证高度正）
        pos = np.random.uniform(-15000, 15000, 3)
        pos[2] = abs(pos[2]) + 500  # 保证 z 为正且不太小（模拟卫星在空中）
        satellites.append(Satellite(sat_id=i, position=pos))
    return satellites


if __name__ == '__main__':
    # 假设用户位置（例如固定在原点）
    user_pos = np.array([0.0, 0.0, 0.0])

    # 生成示例卫星数据（例如 8 颗）
    sats = generate_synthetic_satellites(20)

    n_select = 4  # 所需选择的卫星数量

    # 1. OSSA 算法
    best_combo, dop_ossa = ossa_selection(sats, user_pos, n_select)
    print("OSSA选择结果：", best_combo)
    print("OSSA DGDOP =", dop_ossa)

    # 2. MaxEle 算法
    combo_maxele, dop_maxele = maxele_selection(sats, user_pos, n_select)
    print("\nMaxEle选择结果：", combo_maxele)
    print("MaxEle DGDOP =", dop_maxele)

    # 3. FCSDp 算法
    combo_fcsdp, dop_fcsdp = fcsdp_selection(sats, user_pos, n_select)
    print("\nFCSDp选择结果：", combo_fcsdp)
    print("FCSDp DGDOP =", dop_fcsdp)