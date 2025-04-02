#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例说明：
1. 利用 Skyfield 库加载本地 TLE 文件（例如 "tle.txt"）中保存的卫星 TLE 数据；
2. 定义静态用户位置（例如：[45.75°N, 126.68°E, 100 m]）；
3. 在指定时间范围内（例如 2 分钟、每 5 秒采样一次）获取各颗卫星的位置和速度，
   计算相对于用户的相对位置向量 r 以及对应的误差映射向量（EMV），
   公式：EMV = (r × (r × v)) / |r|³ ；
4. 对单颗卫星在多个时刻得到的 EMV 进行 SVD 拟合，其最小奇异值对应的右奇异向量作为该卫星的几何特征向量（GEV），
   并计算其与理想 90°（水平）的偏差值 |90°-ϕ|（假设 ϕ 取自 GEV 的 z 分量）；
5. 对选取的卫星组成组合（例如每组 4 颗），利用各卫星平均 EMV 构造 Doppler 定位的观测矩阵 G，
   按照公式 (14)：(η·DGDOP)² = tr[(GᵀG)⁻¹]，计算 DGDOP 值（其中 η 根据平均轨道半径计算）；
6. 最后，绘制出“平均 |90°–GEV|”与 DGDOP 的关系曲线图，其中横坐标为 |90°–GEV| 值，纵坐标为 DGDOP 值。
"""

from skyfield.api import load, Topos
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

# ---------------------------
# 1. 加载 TLE 数据
# 请将 TLE 文件（包含标准格式的卫星两行星历数据）保存在路径 "satellite_data\\starlink_tle_20250106_144506.txt"
satellites = load.tle_file('satellite_data\\starlink_tle_20250106_144506.txt')
print('加载了 %d 颗卫星' % len(satellites))
# 为示例起见，选取前 8 颗卫星
selected_sats = satellites[:8]

# ---------------------------
# 2. 定义静态用户位置，例如 [45.75°N, 126.68°E, 100 m]
observer = Topos('45.75 N', '126.68 E', elevation_m=100)

# ---------------------------
# 3. 定义时间采样（例如：2025年1月6日14:00:00起，采样2分钟，每5秒一个采样）
ts = load.timescale()
start_time = ts.utc(2025, 1, 6, 14, 0, 0)
# 采样 0～120 秒，每 5 秒采样一次
step = 5
times = ts.utc(2025, 1, 6, 14, 0, np.arange(0, 121, step))

# ---------------------------
# 4. 定义函数，计算某颗卫星在多个时刻相对于用户的 EMV 样本及平均相对距离
def compute_emv_samples(sat, times):
    # 获得卫星在 times 时刻的三维位置（km）和速度（km/s），将位置单位转换为 m
    sat_pos = sat.at(times).position.km * 1000  # (3, N)
    sat_vel = sat.at(times).velocity.km_per_s * 1000  # 转换为 m/s
    # 用户位置
    obs_pos = observer.at(times).position.km * 1000  # (3, N)
    # 计算相对位置向量 r (satellite - observer)
    r = sat_pos - obs_pos  # (3, N)
    # 计算每个时刻 r 的模长
    r_norm = np.linalg.norm(r, axis=0)
    emv_samples = []
    # 对每个采样时刻逐个计算 EMV
    for i in range(r.shape[1]):
        r_i = r[:, i]
        v_i = sat_vel[:, i]
        # 计算 r × v
        cross1 = np.cross(r_i, v_i)
        # 计算 r × (r × v)
        cross2 = np.cross(r_i, cross1)
        # 计算 EMV = cross2 / |r|³, 注意加上微小量防止除零
        emv_i = cross2 / (np.linalg.norm(r_i)**3 + 1e-9)
        emv_samples.append(emv_i)
    emv_samples = np.array(emv_samples)  # shape (N, 3)
    # 返回 EMV 样本以及该卫星的平均相对距离（用于后续 η 计算）
    mean_r = np.mean(r_norm)
    return emv_samples, mean_r

# ---------------------------
# 5. 对每颗选取的卫星，计算 EMV 样本、利用 SVD 得到 GEV 及与理想 90° 的偏差
sat_data = []  # 存储每颗卫星的相关数据
for sat in selected_sats:
    emv_samples, mean_r = compute_emv_samples(sat, times)
    # 对 EMV 样本矩阵做 SVD，最小奇异值对应的右奇异向量即为平面法向量，即 GEV
    U, S, Vt = np.linalg.svd(emv_samples, full_matrices=False)
    gev = Vt[-1, :]
    gev = gev / np.linalg.norm(gev)
    # 假设理想状态下 GEV 的俯仰角为 90°（即水平方向），这里取 GEV 的 z 分量计算俯仰角：ϕ = arccos(n_z)
    phi_rad = np.arccos(gev[2])
    phi_deg = np.degrees(phi_rad)
    deviation = abs(90 - phi_deg)  # |90°-ϕ|
    # 同时计算该卫星的平均 EMV，后续用于构造观测矩阵
    mean_emv = np.mean(emv_samples, axis=0)
    sat_data.append({
        'sat': sat,
        'gev': gev,
        'phi_deg': phi_deg,
        'deviation': deviation,
        'mean_emv': mean_emv,
        'mean_r': mean_r,
    })
    print("卫星 %s: GEV 俯仰角 = %.2f°, |90°-GEV| = %.2f°" % (sat.name, phi_deg, deviation))

# ---------------------------
# 6. 对选取的卫星按组分组（例如每组 4 颗），构造 Doppler 观测矩阵并计算 DGDOP
# 根据论文，构造观测矩阵 G 的每一行为： [mean_emv / η, 1]
# 此处 η 按组内卫星平均轨道距离计算：
# η = (1/mean_r) * (1 - re/mean_r) * sqrt(GM/mean_r)
GM = 3.986004418e14  # 地球引力常数, m^3/s^2
re = 6371e3         # 地球半径, m
group_size = 4
num_groups = len(sat_data) // group_size
group_results = []  # 存储每组的 DGDOP 与平均 |90°-GEV| 值

for i in range(num_groups):
    group = sat_data[i*group_size : (i+1)*group_size]
    # 计算该组卫星的平均轨道距离
    mean_rs = [item['mean_r'] for item in group]
    group_mean_r = np.mean(mean_rs)
    # 计算 η
    eta = (1.0 / group_mean_r) * (1 - re / group_mean_r) * np.sqrt(GM / group_mean_r)
    # 构造观测矩阵 G (尺寸: group_size x 4)
    G_rows = []
    deviations = []
    for item in group:
        row = np.hstack((item['mean_emv'] / eta, [1]))
        G_rows.append(row)
        deviations.append(item['deviation'])
    G = np.array(G_rows)
    # 依据公式 (14): (η·DGDOP)² = tr[(GᵀG)⁻¹]  →  DGDOP = sqrt(tr(inv(GᵀG)))/η
    GTG = np.dot(G.T, G)
    try:
        inv_GTG = np.linalg.inv(GTG)
        dgdop = np.sqrt(np.trace(inv_GTG)) / eta
    except np.linalg.LinAlgError:
        dgdop = np.nan
    avg_deviation = np.mean(deviations)
    group_results.append({'group_index': i+1, 'dgdop': dgdop, 'avg_deviation': avg_deviation})
    print("第 %d 组: DGDOP = %.2f, 平均 |90°-GEV| = %.2f°" % (i+1, dgdop, avg_deviation))

# ---------------------------
# 7. 修改绘图：横坐标为平均 |90°–GEV| (°)，纵坐标为 DGDOP 值
deviation_vals = [res['avg_deviation'] for res in group_results]
dgdop_vals = [res['dgdop'] for res in group_results]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(deviation_vals, dgdop_vals, color='tab:blue', marker='o', linewidth=2, label='DGDOP')
ax.set_xlabel('|90°–GEV| (°)', fontsize=12)
ax.set_ylabel('DGDOP', fontsize=12)
ax.set_title('GEV 与 DGDOP 的关系曲线', fontsize=14)
ax.grid(True)
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()