import numpy as np
import matplotlib.pyplot as plt

from skyfield.api import load, wgs84

# skyfield 内部也基于 sgp4 做轨道传播；无需手动导入 sgp4, skyfield 已自动集成。

# ========== 1. 读取本地 TLE 文件并筛选指定卫星 ==========
# 请将此文件名修改为你自己的本地 TLE 文件路径
TLE_FILENAME = 'satellite_data\\starlink_tle_20241219_145248.txt'

with open(TLE_FILENAME, 'r') as f:
    lines = f.readlines()

# 假设文件中 TLE 每卫星占 2 行或 3 行(第一行为卫星名字+后面两行元素)
# skyfield 可以用 load.tle_file() 简化此操作。这里演示更底层的方式:
# 若文件格式是标准 3 行(名字 + 两行 TLE) 或 2 行(仅 TLE)，可根据具体情况修改。
# 为简单起见，这里用 load.tle_file 读取后，再选出前 5 颗卫星。

satellites = load.tle_file(TLE_FILENAME)
# 若仅想要前 5 颗，或通过名字筛选:
selected_sats = satellites[:5]  # 假设只取前 5 颗

# ========== 2. 计算星下点轨迹并绘图演示 ==========
ts = load.timescale()
# 设定要绘制的时间区间（例如从当前时刻起的 0~30 分钟，每隔 1 分钟采样）
t0 = ts.now()
minutes_range = np.linspace(0, 30, 31)  # 0~30 分钟，步长 1
time_array = t0 + minutes_range * 60.0 / (24.0 * 60.0)  # skyfield 时间以儒略日为单位

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.set_title("Sub-Satellite Tracks (示例)")

# 简单画一个地理坐标背景(若需要更逼真的地图可用 cartopy 等)
# 此处仅画出经、纬线作为示意
world_lats = np.arange(-90, 91, 30)
world_lons = np.arange(-180, 181, 60)
for lat in world_lats:
    ax.plot([-180, 180], [lat, lat], color='lightgray', linewidth=0.5)
for lon in world_lons:
    ax.plot([lon, lon], [-90, 90], color='lightgray', linewidth=0.5)

ax.set_xlim([-180, 180])
ax.set_ylim([-90, 90])
ax.set_xlabel("Longitude (deg)")
ax.set_ylabel("Latitude (deg)")

# 将 5 颗卫星各自的星下点连接起来
colors = ['r', 'g', 'b', 'c', 'm']
for i, sat in enumerate(selected_sats):
    longs = []
    lats = []
    for t in time_array:
        # 使用 skyfield 计算卫星位置
        geocentric = sat.at(t)
        # 转为地理坐标(经度、纬度、高度)
        subpoint = wgs84.subpoint(geocentric)
        lats.append(subpoint.latitude.degrees)
        longs.append(subpoint.longitude.degrees)

    ax.plot(longs, lats, color=colors[i % len(colors)],
            label=f"{sat.name}")

ax.legend()
plt.show()

# # ========== 3. 多普勒定位方程组示例 ==========
# # 假设我们已有来自这 5 颗卫星在若干时刻的多普勒观测值(由频域算法提取)。
# # 这里演示仅取某一时刻的观测(或多历元也可)，并简化到 2D + 本地钟偏的情况。
#
# # 在真实应用中，这些观测应来自测量： fD_obs[i], i=1..5
# # 这里只示例常量/随机，以演示方程构造。
# fD_obs_example = np.array([500.0, 620.0, 430.0, 570.0, 490.0])  # Placeholder
#
#
# # 需要同一时刻卫星的位置、速度:
# def get_sat_info(satellite, obs_time):
#     """
#     返回卫星在 obs_time 时刻的 [rs, vs]
#     rs = np.array([x_s, y_s, z_s]) (单位: m)
#     vs = np.array([vx_s, vy_s, vz_s]) (单位: m/s)
#     """
#     # 注：skyfield 的 .position.km 以 km 为单位，需要转换为 m
#     geocentric = satellite.at(obs_time)
#     # ITRS (地球惯性/准惯性坐标系下的 X,Y,Z, km)
#     sat_pos = geocentric.position.km * 1000.0
#     sat_vel = geocentric.velocity.km_per_s * 1000.0
#     return sat_pos, sat_vel
#
#
# # 光速
# C = 3e8
# # 若我们统一用相同载波频率(以文章为例: ~11.325 GHz 或 11.575 GHz)
# f_carrier = 11.325e9
#
#
# # 假设接收机高度已知 z=0(海平面)，仅需要求 x_u, y_u, 以及接收机钟偏 f_u
# # 在文章中公式(3)~(12) 的简化版本:
# #   fD_model_i = ( (vᵢ · (r_sᵢ - r_u)) / |r_sᵢ - r_u| ) * (f_carrier / C ) + f_u
# #
# # 残差: resᵢ = fD_obsᵢ - fD_modelᵢ
#
# def residual_2d_f(uxy, sat_pos, sat_vel, fd_obs):
#     """
#     uxy = [x_u, y_u, f_u]
#     sat_pos, sat_vel: 卫星在同一历元的位置/速度(单位 m, m/s)
#     fd_obs: 卫星的多普勒观测值
#     返回: 残差向量 res (长度 = 卫星数)
#     """
#     x_u, y_u, f_u = uxy
#     z_u = 0.0  # 高程辅助(假设海平面)
#
#     res = []
#     for i in range(len(sat_pos)):
#         rs_i = sat_pos[i]  # [x_s, y_s, z_s]
#         vs_i = sat_vel[i]  # [vx, vy, vz]
#         obs_i = fd_obs[i]  # 实测多普勒
#
#         ru = np.array([x_u, y_u, z_u])
#         rsu = rs_i - ru
#         norm_rsu = np.linalg.norm(rsu)
#
#         radial_speed = np.dot(vs_i, rsu) / norm_rsu
#         fd_model_i = radial_speed * (f_carrier / C) + f_u
#
#         res_i = obs_i - fd_model_i
#         res.append(res_i)
#     return np.array(res)
#
#
# def jacobian_2d_f(uxy, sat_pos, sat_vel, fd_obs, eps=1e-6):
#     """
#     使用数值法计算残差对 [x_u, y_u, f_u] 的偏导，形成雅可比矩阵
#     返回 shape = (卫星数, 3)
#     """
#     base_res = residual_2d_f(uxy, sat_pos, sat_vel, fd_obs)
#     J = []
#     for param_idx in range(len(uxy)):
#         perturbed = np.copy(uxy)
#         perturbed[param_idx] += eps
#         new_res = residual_2d_f(perturbed, sat_pos, sat_vel, fd_obs)
#         diff = (new_res - base_res) / eps
#         # diff shape = (卫星数,)
#         J.append(diff)
#     # 拼成 (卫星数, 3) 矩阵
#     return np.stack(J, axis=1)
#
#
# def gauss_newton_2d_f(uxy0, sat_pos, sat_vel, fd_obs, max_iter=20, tol=1e-5):
#     """
#     高斯-牛顿迭代，解 2D + f_u
#     sat_pos, sat_vel: 每颗卫星在同一时刻的 [pos, vel]
#     fd_obs: 观测多普勒
#     """
#     x_est = np.copy(uxy0)
#     for it in range(max_iter):
#         r = residual_2d_f(x_est, sat_pos, sat_vel, fd_obs)  # (N,)
#         J = jacobian_2d_f(x_est, sat_pos, sat_vel, fd_obs)  # (N, 3)
#
#         # 最小二乘修正量 delta = (J^T J)^(-1) * J^T * r
#         JTJ = J.T @ J
#         JTr = J.T @ r
#         delta = np.linalg.inv(JTJ) @ JTr
#
#         x_new = x_est + delta
#         if np.linalg.norm(delta) < tol:
#             x_est = x_new
#             break
#         x_est = x_new
#
#     final_res = np.linalg.norm(residual_2d_f(x_est, sat_pos, sat_vel, fd_obs))
#     return x_est, final_res
#
#
# # ========== 举例：在目标时刻对 5 颗卫星做一次定位 ==========
# # 需要真实多普勒观测值，这里只能用占位符 fd_obs_example
# # 请用从实际观测中提取的多普勒频率替换
# obs_time = t0  # 演示：在当前时刻
# positions = []
# velocities = []
#
# for sat in selected_sats:
#     p, v = get_sat_info(sat, obs_time)
#     positions.append(p)
#     velocities.append(v)
#
# positions = np.array(positions)  # shape = (5, 3)
# velocities = np.array(velocities)  # shape = (5, 3)
#
# # 初始猜测(例如在地心附近... 实际中应给出更合理的先验，比如已知在地表某处)
# # x_u, y_u ~ 0, f_u ~ 0
# init_guess = np.array([0.0, 0.0, 0.0])
#
# solution, residual = gauss_newton_2d_f(init_guess, positions, velocities,
#                                        fD_obs_example,
#                                        max_iter=20, tol=1e-6)
#
# print("===== 多普勒定位解算(示例) =====")
# print(f"解算结果: x_u={solution[0]:.2f} m, y_u={solution[1]:.2f} m, f_u={solution[2]:.2f} Hz")
# print(f"残差范数: {residual:.4f}")