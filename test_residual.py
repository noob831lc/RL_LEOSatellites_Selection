import numpy as np
from scipy.optimize import least_squares

# ----------------------------------------------------
# 输入: 卫星位置、速度、多普勒观测值、中心频率等
# ----------------------------------------------------
from skyfield.toposlib import wgs84

N = 4  # 示例：观测到 4 颗星
c = 3e8  # 光速 (m/s)
f_c = 11.325e9  # 选取星链导频的一种中心频率 (单位: Hz)

# ECEF坐标系下 卫星位置 (x, y, z)，已通过TLE+SGP4等方法计算得到，单位: m
r_s = np.array([
    [1.2e6, 2.1e6, 2.1e6],
    [2.2e6, 2.8e6, 2.6e6],
    [3.1e6, 1.2e6, 3.0e6],
    [2.1e6, 3.2e6, 3.6e6]
])

# 卫星速度 (vx, vy, vz)，单位: m/s
v_s = np.array([
    [2500.0, 2200.0, 700.0],
    [2100.0, 2350.0, 650.0],
    [2200.0, 2500.0, 800.0],
    [2400.0, 2100.0, 1000.0]
])

# 实际观测到的多普勒值 (统计/提取后), 单位: Hz
fD_obs = np.array([
    -12345.6,
    -11300.2,
    -12580.9,
    -13210.0
])

# 假设接收机(用户)的海拔已知为 h0，考虑地球曲率则需要更复杂模型
# 下面先演示( x, y )未知，高度 z0 固定:
z0 = 0.0  # 简化假设


# ----------------------------------------------------
# 2) 构建残差函数
#    x_vec = [ x, y, f_u, f_S1, f_S2, ..., f_SN ] (共 N+3 个量)
# ----------------------------------------------------
def doppler_residuals(x_vec):
    """
    x_vec: [ x, y, f_u, f_S1, ..., f_SN ]
    返回: 对应 N 颗星的残差向量 [d1, d2, ..., dN]
    """
    # 解析未知量
    x_user = x_vec[0]
    y_user = x_vec[1]
    f_u = x_vec[2]  # 接收机频率偏差
    f_S = x_vec[3:]  # 卫星频率偏差数组, 长度 N

    # 计算每颗星的理论多普勒
    residuals = []
    for i in range(N):
        # 卫星位置速度
        r_si = r_s[i]
        v_si = v_s[i]
        # 卫星钟差
        f_Si = f_S[i]

        # 接收机位置向量
        r_u = np.array([x_user, y_user, z0])

        # 几何关系项
        # rs - ru
        diff_pos = r_si - r_u
        norm_diff = np.linalg.norm(diff_pos)  # ||r_s - r_u||

        # vs · (rs - ru)
        dot_val = np.dot(v_si, diff_pos)

        # 多普勒公式 (忽略 +/- 号差异，只示意)
        # fD_theo = (dot_val / norm_diff)*(f_c/c) + f_Si + f_u
        fD_theo = (dot_val / norm_diff) * (f_c / c) + f_Si + f_u

        # 残差 = 实测 - 理论
        r_i = fD_obs[i] - fD_theo
        residuals.append(r_i)
    return np.array(residuals)


# ----------------------------------------------------
# 3) 设置初值并调用数值优化求解
# ----------------------------------------------------
# 初始猜测 (x, y), 单位 m
x0_guess = 1e3
y0_guess = 1e3
f_u_guess = 0.0
f_S_guess = [0.0] * N  # 每一颗卫星的频偏初值都先设为0

x_init = np.array([x0_guess, y0_guess, f_u_guess] + f_S_guess)

res = least_squares(doppler_residuals, x_init, method='trf')  # Levenberg-Marquardt

# ----------------------------------------------------
# 4) 输出结果
# ----------------------------------------------------
x_est = res.x[0]  # x
y_est = res.x[1]  # y
f_u_est = res.x[2]  # 接收机钟差
f_S_est = res.x[3:]  # 每颗星钟差
position = wgs84.latlon(x_est, y_est, 0)
# 获取纬度、经度(度)和高度(米)
lat = position.latitude.radians
lon = position.longitude.radians
height = position.elevation.m
print("优化结束：", res.message)
print("迭代总次数：", res.nfev)
print(f"接收机估计位置：x={x_est:.3f} m, y={y_est:.3f} m, 高度为 z={z0} m")
print(f"接收机估计位置：lat={lat:.3f}°, lon={lon:.3f}°, 高度为 height={height} m")
print(f"接收机频率偏差估计：f_u={f_u_est:.6f} Hz")
for i in range(N):
    print(f"第{i + 1}颗卫星的载波频偏估计：f_S{i + 1}={f_S_est[i]:.6f} Hz")

print("残差向量：", res.fun)
print("残差范数：", np.linalg.norm(res.fun))
