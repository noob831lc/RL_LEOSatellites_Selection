import itertools
import math
import numpy as np
from typing import List, Dict, Tuple

############################################################################
# 模拟输入：假定我们已经有 M 颗可见卫星，每颗卫星的数据包括其在当前时刻的空间位置 pos (x,y,z)
# 以及速度 vel (vx,vy,vz)，以下仅是示例数据结构与初始化
############################################################################

# 演示用的简单数据，M=8 颗星，用户自行替换为真实星历数据
# 每个卫星字典包含 'pos':(x,y,z), 'vel':(vx,vy,vz)
satellite_data = [
    {'id': i,
     'pos': (6000.0 + 100 * i, 1000.0 + 50 * i, 5000.0 + 70 * i),  # 单位km或m自行决定，这里仅演示
     'vel': (0.0, 7.5 + i * 0.1, 0.0)}  # 简化: 仅给y方向速度，以演示
    for i in range(20)
]

# 用户位置(示例) (x,y,z)
user_pos = (6378.0, 0.0, 0.0)  # 假设地球半径约6378km，用户在地面(仅示意)
N = 4  # 要选取的卫星数


############################################################################
# 工具函数：计算多普勒观测矩阵 G_{V_d} 以及 DGDOP
# （以下函数为简化示例，真实应用需结合多历元、多普勒频差、钟差等）
############################################################################

def calc_unit_vector(sat_pos: np.ndarray, user_pos: np.ndarray) -> np.ndarray:
    """
    返回用户 -> 卫星 的单位向量 e = (sat_pos - user_pos)/|sat_pos - user_pos|
    """
    vec = sat_pos - user_pos
    dist = np.linalg.norm(vec)
    if dist < 1e-12:
        # 避免除零，仅演示
        return np.zeros(3)
    return vec / dist


def calc_doppler_emv(sat_pos: np.ndarray, sat_vel: np.ndarray, user_pos: np.ndarray) -> np.ndarray:
    """
    计算对单历元载波多普勒的误差映射向量 (e_x, e_y, e_z),
    这里按照论文(8)-(9)式的思路做个简单的近似: e ~ (r_hat x (r_hat x v_sat))/|r|
    注：此处仅做最简近似，用于展示流程
    """
    r_vec = sat_pos - user_pos  # 用户->卫星向量
    r_hat = r_vec / (np.linalg.norm(r_vec) + 1e-16)
    # v_sat 近似不考虑用户速度(例如用户静止)
    cross_1 = np.cross(r_hat, sat_vel)
    cross_2 = np.cross(r_hat, cross_1)
    # 作归一化：与论文公式类似
    return cross_2 / (np.linalg.norm(r_vec) + 1e-16)


def calc_G_matrix(satellite_subset: List[Dict], user_pos: Tuple[float, float, float]) -> np.ndarray:
    """
    对选出的卫星子集计算 G 矩阵 (以单历元多普勒为例, 大小 n x 4, 其中 n = 组合内卫星数)
    G 矩阵每行对应 [ e_x, e_y, e_z, 1 ], 详见论文(8)等
    """
    rows = []
    user_arr = np.array(user_pos, dtype=float)
    for sat in satellite_subset:
        sat_pos = np.array(sat['pos'], dtype=float)
        sat_vel = np.array(sat['vel'], dtype=float)
        e_vec = calc_doppler_emv(sat_pos, sat_vel, user_arr)
        # [ex, ey, ez, 1]
        row = np.hstack((e_vec, [1.0]))
        rows.append(row)
    G = np.array(rows, dtype=float)
    return G


def calc_dgdop(satellite_subset: List[Dict], user_pos: Tuple[float, float, float], eta=1.0) -> float:
    """
    计算一组卫星对应的 DGDOP
    DGDOP = (1/eta) * sqrt( tr( (G^T G)^-1 ) )
    eta 缩放因子(论文(11))此处默认为1仅做演示
    """
    G = calc_G_matrix(satellite_subset, user_pos)
    # 若卫星数不足4，可能导致(G^T G)不可逆，做个简单健壮处理
    GTG = G.T @ G
    # 逆矩阵
    try:
        inv_GTG = np.linalg.inv(GTG)
        tr_val = abs(np.trace(inv_GTG))
        dgdop_val = (1.0 / eta) * math.sqrt(tr_val)
    except np.linalg.LinAlgError:
        # 不可逆情况，返回一个大值
        dgdop_val = 1e10
    return dgdop_val


############################################################################
# 1. OSSA（遍历搜索）实现
#    从 M 颗卫星中选 N 颗，计算每个组合的DGDOP，返回DGDOP最小的组合
############################################################################

def select_satellites_OSSA(sat_data: List[Dict], user_pos: Tuple[float, float, float], N: int) -> List[Dict]:
    """
    OSSA: 完全遍历所有 C(M,N) 组合，找出 DGDOP 最优(最小)的卫星组合
    """
    best_combination = None
    best_dgdop = float('inf')

    for subset_indices in itertools.combinations(range(len(sat_data)), N):
        subset = [sat_data[i] for i in subset_indices]
        dgdop_val = calc_dgdop(subset, user_pos, eta=1.0)
        if dgdop_val < best_dgdop:
            best_dgdop = dgdop_val
            best_combination = subset

    return best_combination


############################################################################
# 2. MaxEle（最大仰角选星）
#    简单地先计算每颗卫星相对于用户位置的仰角，然后选取前N个仰角最高的卫星
############################################################################

def calc_elevation(sat_pos: np.ndarray, user_pos: np.ndarray) -> float:
    """
    计算卫星相对于用户的仰角(简化):
      elevation = arcsin( (sat_pos - user_pos).normalize() · U_z )
    假设用户在地面, U_z ~ (x,y,z)方向上z为法线(这里仅做示例).
    在真实情况须先将 user_pos -> ECEF/ENU 转换,然后再求仰角
    """
    # 简化: 假设用户在赤道(6378,0,0),则 "Up" 方向可近似 (1,0,0) 在本地坐标
    # 这是非常粗略的简化，仅供演示.
    user_arr = np.array(user_pos, dtype=float)
    sat_vec = sat_pos - user_arr
    dist = np.linalg.norm(sat_vec)
    if dist < 1e-12:
        return -999.0
    # 本例假设 user_pos ~ (6378,0,0) -> Up方向= (1,0,0) in local sense
    up_dir = np.array([1.0, 0.0, 0.0])
    sat_unit = sat_vec / dist
    dot_val = np.dot(sat_unit, up_dir)
    # arcsin( dot_val )  范围 [-pi/2, pi/2]
    elev = math.degrees(math.asin(dot_val))
    return elev


def select_satellites_MaxEle(sat_data: List[Dict], user_pos: Tuple[float, float, float], N: int) -> List[Dict]:
    """
    传统最大仰角选星：先计算每颗卫星的仰角，按从大到小排序，取前N颗
    """
    sat_enumerated = []
    for sat in sat_data:
        sat_pos = np.array(sat['pos'], dtype=float)
        elv = calc_elevation(sat_pos, user_pos)
        sat_enumerated.append((sat, elv))
    # 按仰角降序
    sat_enumerated.sort(key=lambda x: x[1], reverse=True)
    selected = [x[0] for x in sat_enumerated[:N]]
    return selected


############################################################################
# 3. FCSDp（快速聚类选星）
#    基于“几何特征向量(GEV)+分层聚类+匈牙利算法(可选)”的思路。
#    演示中只实现最核心的“GEV计算+分层聚类(K-means++)”来选星。
#    匈牙利算法部分在多时间片场景下使用，此处仅示例单时刻。
############################################################################

def calc_gev_for_satellite(sat: Dict, user_pos: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    对单颗卫星计算其 GEV (几何特征向量) 的极角表示 (phi, beta)
    此处为“简化版”，用一次性多普勒EMV近似。
    在真实多历元场景下，可对多历元EMV取平均或其他方式综合。
    """
    user_arr = np.array(user_pos)
    sat_pos = np.array(sat['pos'], dtype=float)
    sat_vel = np.array(sat['vel'], dtype=float)

    # 示例：取一次性EMV作为GEV
    e_vec = calc_doppler_emv(sat_pos, sat_vel, user_arr)
    norm_e = np.linalg.norm(e_vec)
    if norm_e < 1e-12:
        return (0.0, 0.0)

    # 归一化
    e_unit = e_vec / norm_e
    # 极角: phi = 与Z轴的夹角? or 与X轴夹角? 论文中可定义不同.
    # 这里简单定义:
    #   phi   = angle to "vertical" (类似与Z方向)        in range [0, pi]
    #   beta  = azimuth angle in horizontal plane       in range [-pi, pi]
    # 仅做演示，与文中完全对应需要视坐标系统而定
    # 下方以Z轴 = (0,0,1)为参考
    ez = np.array([0, 0, 1.0])
    phi = math.acos(np.dot(e_unit, ez))  # 0=向上, pi=向下
    # beta：投影到XY平面后的方位角
    proj_xy = e_unit.copy()
    proj_xy[2] = 0
    len_xy = np.linalg.norm(proj_xy)
    if len_xy < 1e-9:
        beta = 0.0
    else:
        beta = math.atan2(proj_xy[1], proj_xy[0])

    return (phi, beta)


def kmeans_plus_plus_init(data_points: List[Tuple[float]], k: int) -> List[Tuple[float]]:
    """
    简易K-means++ 初始质心选取
    data_points: [(beta1,), (beta2,), ...]
    k: 多少个聚类
    返回初始聚类中心列表
    """
    import random
    centers = []
    centers.append(data_points[random.randint(0, len(data_points) - 1)])
    while len(centers) < k:
        # 计算每个点与最近中心的距离
        dist_arr = []
        for p in data_points:
            min_d = min([abs(p[0] - c[0]) for c in centers])
            dist_arr.append(min_d)
        # 轮盘赌
        s = sum(dist_arr)
        r = random.random() * s
        cumsum = 0.0
        for idx, d in enumerate(dist_arr):
            cumsum += d
            if cumsum >= r:
                centers.append(data_points[idx])
                break
    return centers


def run_kmeans(data_points: List[Tuple[float]], k: int,
               max_iter=50) -> List[List[Tuple[float]]]:
    """
    仅针对一维数据 points = [(beta,), ...] 做 K-means
    返回聚类好的 k 个簇，每个簇是一些点的列表
    """
    # 初始化质心
    centers = kmeans_plus_plus_init(data_points, k)
    for _ in range(max_iter):
        # 1) Assign
        clusters = [[] for __ in range(k)]
        for p in data_points:
            dlist = [abs(p[0] - c[0]) for c in centers]
            idx_min = np.argmin(dlist)
            clusters[idx_min].append(p)
        # 2) Update center
        new_centers = []
        for cl in clusters:
            if len(cl) == 0:
                new_centers.append((0.0,))
            else:
                mean_beta = np.mean([x[0] for x in cl])
                new_centers.append((mean_beta,))
        # check converge
        shift = sum(abs(new_centers[i][0] - centers[i][0]) for i in range(k))
        centers = new_centers
        if shift < 1e-6:
            break
    return clusters


def select_satellites_FCSDp(sat_data: List[Dict], user_pos: Tuple[float, float, float], N: int) -> List[Dict]:
    """
    FCSDp主流程(单时刻简化版)：
      1) 计算每颗卫星的GEV( phi, beta )
      2) 根据 phi > pi/2 分为上下层 C+ / C-
      3) 分别对上层、下层进行 K-means++ (beta为特征)
         - 若N为偶数，上下层各分 N/2 个; 若N为奇数，根据需要分 (N+1)/2 & (N-1)/2
      4) 每个聚类中选一个最优卫星(如与聚类中心距离最小等)
      5) 将上下层结果合并，若结果数量超过N可能需再做一层筛选(此处略)
    """

    # 1) 计算GEV
    sat_gevs = []  # [(idx, phi, beta), ...]
    for i, sat in enumerate(sat_data):
        phi, beta = calc_gev_for_satellite(sat, user_pos)
        sat_gevs.append((i, phi, beta))

    # 2) 上下层划分
    upper_layer = []  # phi> pi/2
    lower_layer = []  # phi<= pi/2
    for item in sat_gevs:
        if item[1] > math.pi / 2.0:
            upper_layer.append(item)
        else:
            lower_layer.append(item)

    # 3) k-means++ 聚类 (beta 维度)
    def cluster_and_select_candidates(data_list, k):
        """
        对 data_list，以 (beta,) 进行K-means++聚类，
        然后每个簇选出与均值最接近的一颗卫星
        """
        if k <= 0 or len(data_list) == 0:
            return []
        # 构造 (beta,) 形式
        data_points = [(x[2],) for x in data_list]  # x=(idx,phi,beta)
        clusters = run_kmeans(data_points, k)
        # 对每个簇，找到原始data里与其中心最近的卫星
        selected_idx = []
        # 由于run_kmeans只返回分组，这里需要重新计算质心
        for cl in clusters:
            if len(cl) == 0:
                continue
            mean_beta = np.mean([p[0] for p in cl])
            # 在data_list里匹配
            best_item = None
            best_dist = float('inf')
            for p in cl:
                # p=(beta,)
                distp = abs(p[0] - mean_beta)
                if distp < best_dist:
                    best_dist = distp
                    best_item = p[0]
            # 找到 best_item 对应的原始卫星
            # 可能有多个beta相同的情况，此处简单选第一个匹配
            for d in data_list:
                if abs(d[2] - best_item) < 1e-9:
                    selected_idx.append(d[0])
                    break
        return list(set(selected_idx))  # 去重

    # 按照N奇偶划分
    if N % 2 == 0:
        k_upper, k_lower = N // 2, N // 2
    else:
        k_upper, k_lower = (N + 1) // 2, (N - 1) // 2

    idx_upper = cluster_and_select_candidates(upper_layer, k_upper)
    idx_lower = cluster_and_select_candidates(lower_layer, k_lower)

    selected_ids = idx_upper + idx_lower
    # 若集合大小超过N，需要做其它裁剪策略;若不够N(例如某层卫星不足),则可补充/回退
    # 此处仅做简单截断/或补充:
    if len(selected_ids) > N:
        selected_ids = selected_ids[:N]
    elif len(selected_ids) < N:
        # 补足: 取尚未被选中的卫星中 DGDOP最优的
        remain = [i for i in range(len(sat_data)) if i not in selected_ids]
        # 简单做个迭代选星
        while len(selected_ids) < N and len(remain) > 0:
            candidate_scores = []
            for r in remain:
                test_subset = [sat_data[x] for x in selected_ids + [r]]
                dgdop_val = calc_dgdop(test_subset, user_pos)
                candidate_scores.append((r, dgdop_val))
            candidate_scores.sort(key=lambda x: x[1])
            best_r = candidate_scores[0][0]
            selected_ids.append(best_r)
            remain.remove(best_r)

    # 构造输出
    selected = [sat_data[i] for i in selected_ids]
    return selected


############################################################################
# 示例：对上述三个算法进行调用与对比
############################################################################
if __name__ == "__main__":
    ossa_result = select_satellites_OSSA(satellite_data, user_pos, N)
    maxele_result = select_satellites_MaxEle(satellite_data, user_pos, N)
    fcsdp_result = select_satellites_FCSDp(satellite_data, user_pos, N)

    print("== OSSA ==")
    print("Selected IDs:", [sat['id'] for sat in ossa_result])
    print("DGDOP:", calc_dgdop(ossa_result, user_pos))

    print("\n== MaxEle ==")
    print("Selected IDs:", [sat['id'] for sat in maxele_result])
    print("DGDOP:", calc_dgdop(maxele_result, user_pos))

    print("\n== FCSDp ==")
    print("Selected IDs:", [sat['id'] for sat in fcsdp_result])
    print("DGDOP:", calc_dgdop(fcsdp_result, user_pos))