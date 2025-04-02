import numpy as np
from scipy.optimize import least_squares
from skyfield.toposlib import wgs84


# 多普勒定位方程组


def starlink_doppler_positioning(
        r_s,
        v_s,
        fD_obs,
        z0=0.0,
        f_c=11.325e9,
        c=3e8,
        x0_guess=0.0,
        y0_guess=0.0,
        f_u_guess=0.0
):
    """
    使用瞬时多普勒定位方法来估计接收机坐标 (x, y) 以及频率偏差 (f_u, f_Si)。

    参数：
    ----------
    r_s : ndarray of shape (N, 3)
        卫星位置向量 (单位：m)，N 颗星，每行 (x_si, y_si, z_si)。
    v_s : ndarray of shape (N, 3)
        卫星速度向量 (单位：m/s)，和 r_s 对应。
    fD_obs : ndarray of shape (N,)
        提取到的各颗卫星的多普勒观测值 (单位：Hz)。
    z0 : float, optional
        接收机高度 (单位：m)，如果只做水平定位，可固定此值。
    f_c : float, optional
        星链导频中心频率，如 11.325 GHz。
    c : float, optional
        光速，默认为 3e8 m/s。
    x0_guess : float, optional
        对 x 坐标的初值猜测。
    y0_guess : float, optional
        对 y 坐标的初值猜测。
    f_u_guess : float, optional
        对接收机频率偏差的初值猜测。

    返回：
    ----------
    result_dict : dict
        结果，包括：
        - x, y : 接收机估计的水平位置 (m)
        - f_u_est : 接收机频率偏差 (Hz)
        - f_s_est : 每颗星的载波频移估计值 (Hz)
        - residuals : 残差向量
        - cost : 残差范数或代价函数
        - success : bool，是否收敛
        - message : 优化器的返回提示信息
    """
    # 卫星数量
    N = r_s.shape[0]
    if v_s.shape[0] != N or fD_obs.shape[0] != N:
        raise ValueError("r_s, v_s, fD_obs 必须长度相同（ N 行 ）！")

    # 构建残差函数
    def doppler_residuals(x_vec):
        """
        x_vec = [ x, y, f_u, f_S1, f_S2, ..., f_SN ] (共 N + 3 个量)
        返回对应 N 颗星的残差 [d1, d2, ..., dN]
        """
        x_user = x_vec[0]
        y_user = x_vec[1]
        f_u = x_vec[2]  # 接收机频率偏差
        f_s = x_vec[3:]  # 每颗星的未知频差

        residuals = []
        for i in range(N):
            # 第 i 颗卫星
            r_si = r_s[i]
            v_si = v_s[i]
            f_si = f_s[i]

            # 构建接收机位置
            r_u = np.array([x_user, y_user, z0])

            # 几何Doppler项
            diff_pos = r_si - r_u
            norm_diff = np.linalg.norm(diff_pos)  # ||r_s - r_u||
            dot_val = np.dot(v_si, diff_pos)  # v_s_i · (r_s_i - r_u)

            # 理论多普勒（此处忽略 ± 号差异，只示意）
            fD_theo = (dot_val / norm_diff) * (f_c / c) + f_si + f_u

            # 残差 = 实测多普勒 - 理论值
            r_i = fD_obs[i] - fD_theo
            residuals.append(r_i)
        return np.array(residuals)

    # 设置初值
    f_s_guess = np.zeros(N)  # 卫星频偏初值都设为 0
    x_init = np.array([x0_guess, y0_guess, f_u_guess] + f_s_guess.tolist())

    # 调用最小二乘求解
    res = least_squares(doppler_residuals, x_init, method='trf')

    # 解析结果
    x_est = res.x[0]  # x
    y_est = res.x[1]  # y
    f_u_est = res.x[2]  # 接收机钟差
    f_s_est = res.x[3:]  # 每颗卫星钟差
    position = wgs84.latlon(x_est, y_est, 0)
    # 获取纬度、经度(度)和高度(米)
    lat = position.latitude.radians
    lon = position.longitude.radians
    height = position.elevation.m
    np.set_printoptions(precision=6, suppress=True, floatmode='fixed', linewidth=100)
    print(f'jac:\n{res.jac}')
    print("优化目标函数最终cost：", res.cost)
    print("是否成功：", res.success)
    print("优化结束：", res.message)
    print("迭代总次数：", res.nfev)
    print(f"接收机估计位置：x={x_est:.3f} m, y={y_est:.3f} m, 高度为 z={z0} m")
    print(f"接收机估计位置：lat={lat:.3f}°, lon={lon:.3f}°, 高度为 height={height} m")
    print(f"接收机频率偏差估计：f_u={f_u_est:.6f} Hz")
    for i in range(N):
        print(f"第{i + 1}颗卫星的载波频偏估计：f_S{i + 1}={f_s_est[i]:.6f} Hz")

    print("残差向量：", res.fun)
    print("残差范数：", np.linalg.norm(res.fun))

    # 打包结果
    result_dict = {
        "x": x_est,
        "y": y_est,
        "z_assumed": z0,
        "f_u_est": f_u_est,
        "f_s_est": f_s_est,
        "residuals": res.fun,
        "cost": np.linalg.norm(res.fun),
        "success": res.success,
        "message": res.message
    }

    return result_dict

# def doppler_residual(params, sat_positions, sat_velocities, Z):
#     """
#     计算多普勒方程组的残差向量:
#       Z_i = v_i . (r_i - x) / ||r_i - x|| + a   (忽略噪声)
#     其中 Z_i = λ * f_i
#
#     params: [x, y, z, a]  --> 待求解参数
#     sat_positions:  shape=(M,3)  每条观测对应的卫星坐标
#     sat_velocities: shape=(M,3)  每条观测对应的卫星速度
#     Z:              shape=(M,)   每条观测的伪距变化率(= λ * f_i)
#
#     返回值:
#       residuals:     shape=(M,)   每条观测方程的残差
#     """
#     x, y, z, a = params
#     # 接收机位置向量
#     rx = np.array([x, y, z])
#
#     # 逐条观测计算理论值
#     # r_i - x
#     diff_vec = sat_positions - rx.reshape((1, 3))  # shape=(M,3)
#     # ||r_i - x||
#     dist = np.linalg.norm(diff_vec, axis=1)  # shape=(M,)
#     # dot( v_i, diff_vec / dist )
#     proj = np.sum(sat_velocities * (diff_vec / dist.reshape(-1, 1)), axis=1)  # shape=(M,)
#
#     # 理论预测量: v_i(...) + a
#     Z_pred = proj + a
#
#     # 残差 = 实测 - 理论
#     residuals = Z - Z_pred
#     return residuals
#
#
# def solve_doppler_leastsq(sat_positions, sat_velocities, Z, x0):
#     """
#     利用最小二乘(非线性)求解多普勒定位方程组.
#
#     sat_positions:  (M,3)  每条观测对应的卫星坐标
#     sat_velocities: (M,3)  每条观测对应的卫星速度
#     Z: (M,)                 每条观测对应的  λ * f_i
#     x0: 初始猜测 [x0, y0, z0, a0]
#
#     return: 最优解, [x, y, z, a]
#     """
#     # 调用scipy.optimize.least_squares
#     result = least_squares(
#         fun=doppler_residual,
#         x0=x0,
#         args=(sat_positions, sat_velocities, Z),
#         method='lm'  # or 'trf', 'dogbox' etc. 按需选择
#     )
#     return result.x, result.cost, result.success
#
#
# # ---------------- 以下为示例使用 ----------------
# if __name__ == "__main__":
#     # 假设我们总共观测了 M=6 条(或更多)数据:
#     M = 6
#
#     # 示例: 每条观测的卫星位置 sat_positions[m] = [x_s, y_s, z_s]
#     # 这通常由TLE + SGP4推算得到, 以下仅为示例数
#     sat_positions = np.array([
#         [1.2e6, 2.4e6, 6.3e6],
#         [1.4e6, -2.3e6, 6.2e6],
#         [1.6e6, 3.1e6, 5.9e6],
#         [2.0e6, 1.1e6, 6.0e6],
#         [2.2e6, -1.5e6, 5.8e6],
#         [1.9e6, 2.0e6, 6.1e6],
#     ])
#
#     # 每条观测的卫星速度 sat_velocities[m] = [vx, vy, vz]
#     sat_velocities = np.array([
#         [-1000., 2700., 1000.],
#         [1600., 2100., 1200.],
#         [1000., -2800., 1500.],
#         [-1200., 2000., 1400.],
#         [1300., 2600., 1500.],
#         [1000., -2000., 1300.],
#     ])
#
#     # 给定M条多普勒观测的 Z = λ * f_i, 同样为示例
#     Z = np.array([3300., 4100., 2800., 3600., 4200., 3000.])
#
#     # 设置初始猜测 [x0, y0, z0, a0]
#     x0 = np.array([0.0, 0.0, 0.0, 0.0])
#
#     # 调用求解
#     x_opt, cost, success = solve_doppler_leastsq(sat_positions, sat_velocities, Z, x0)
#
#     print("定位求解结果：", x_opt)
#     print("优化目标函数最终cost：", cost)
#     print("是否成功：", success)
