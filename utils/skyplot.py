import numpy as np
from datetime import datetime, timedelta, timezone
from skyfield.api import load
import matplotlib.pyplot as plt
from skyfield.framelib import itrs, ICRS, tirs, ICRS_to_J2000
from skyfield.toposlib import Topos


# 绘图相关的函数


# 绘制星下点轨迹图 参数 起始时间 观测时间长度(单位 s)  观测卫星列表 观测点经纬度
def sky_plot(
    start_time_utc,
    time_list_len,
    satellites,
    lat, lon
):
    ts = load.timescale()
    time_list = [ts.from_datetime(start_time_utc + timedelta(seconds=s)) for s in range(time_list_len)]

    # 创建存储 ECEF 坐标的数组
    sat_ecef = np.zeros((time_list_len, len(satellites), 3))

    # 地球参数
    a = 6378.137  # 地球长半轴（赤道半径，单位：km）
    e = 0.0818191908426  # 地球偏心率

    # # 遍历所有时间和卫星，计算 ECEF 坐标
    for i, t in enumerate(time_list):
        for j, sat in enumerate(satellites):
            geocentric = sat.at(t)

            # 获取纬度、经度、高度
            pos = geocentric.frame_xyz(itrs)
            x, y, z = pos.m
            # 转换为 ECEF 坐标
            # 存储结果
            sat_ecef[i, j, 0] = x
            sat_ecef[i, j, 1] = y
            sat_ecef[i, j, 2] = z
    #
    phi = np.deg2rad(lat)  # 观测者纬度（示例：30°）
    lam = np.deg2rad(lon)  # 观测者经度（示例：120°）
    ts = load.timescale()

    # 创建地面站点对象
    location = Topos(latitude_degrees=lat,
                     longitude_degrees=lon,
                     elevation_m=100)

    # 获取 ECEF 坐标
    geocentric = location
    position = geocentric.at(ts.from_datetime(start_time_utc))

    # 提取 ECEF 坐标 (单位: 米)

    Xr, Yr, Zr = position.frame_xyz(itrs).m

    R = get_rotation_matrix(phi, lam)

    az_by_sv = []  # list of arrays，az_by_sv[s] 存储第 s 颗卫星所有时刻的方位角
    el_by_sv = []  # list of arrays
    #
    for s in range(len(satellites)):
        az_list = []
        el_list = []
        for t in range(time_list_len):
            V = sat_ecef[t, s, :] - np.array([Xr, Yr, Zr])
            ENU = R.dot(V)

            E_ = ENU[0]
            N_ = ENU[1]
            U_ = ENU[2]

            # 计算方位角[0, 2π)
            az = np.arctan2(E_, N_)
            if az < 0:
                az += 2 * np.pi

            # 计算仰角
            horiz_dist = np.sqrt(E_ ** 2 + N_ ** 2)
            el = np.arctan2(U_, horiz_dist)

            az_list.append(az)
            el_list.append(el)

        az_by_sv.append(np.array(az_list))
        el_by_sv.append(np.array(el_list))

    # ----------------------------
    # 4. Skyplot 绘制
    # ----------------------------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    ax.set_theta_zero_location('N')  # 0度在北方（顶部）
    ax.set_theta_direction(-1)  # 顺时针方向增加

    # 设置径向范围（0-90度）- 表示天顶角
    ax.set_rlim(0, 90)

    # 设置方位角刻度标签（每30度）
    angles_deg = np.arange(0, 360, 30)
    ax.set_thetagrids(angles_deg, labels=[f"{ang}°" for ang in angles_deg])

    # 设置径向网格和标签 - 关键修改部分
    # 我们需要在30°, 60°, 90°位置设置同心圆
    radii = [30, 60, 90]  # 这些值直接对应天顶角（90-仰角）

    for r in radii:
        circle = plt.Circle((0, 0), r, transform=ax.transData._b,
                            fill=False, edgecolor='gray', linestyle='-', alpha=0.5)
        ax.add_artist(circle)

        # 手动添加标签（与图片相同位置）
        # 在90度方位角（东方）位置添加标签
    ax.text(np.radians(90), 30, "30°", ha='left', va='center', fontsize=9)
    ax.text(np.radians(90), 60, "60°", ha='left', va='center', fontsize=9)
    ax.text(np.radians(90), 90, "90°", ha='left', va='center', fontsize=9)

    sat_nums = len(satellites)
    for s in range(sat_nums):
        # 将仰角从弧度 -> 度，计算 zenith angle = 90 - elevation(deg)
        az_deg = np.degrees(az_by_sv[s])
        el_deg = np.degrees(el_by_sv[s])
        zen_deg = 90.0 - el_deg
        # 转成绘图用的单位
        az_rad = np.radians(az_deg)
        zen_rad = np.radians(zen_deg)

        # ax.plot(
        #     az_rad,
        #     zen_rad,
        #     label=f"SV {s + 1}"
        # )
        ax.plot(az_rad, zen_rad, linestyle='dotted', linewidth=2.5, label=f"{satellites[s].name}")

    # 设置表面参数
    ax.set_rmax(np.radians(90))
    ax.set_thetagrids(range(0, 360, 30))
    ax.set_rlabel_position(180)

    rad_labels = [f"{90 - angle}°" for angle in [90, 60, 30, 0]]

    # ax.set_yticklabels(rad_labels)
    ax.set_rticks([np.radians(i) for i in [0, 30, 60, 90]])

    # 设置网格线样式
    ax.grid(True, color='gray', linestyle='-', linewidth=1)

    ax.set_title("Skyplot", va='bottom')
    ax.grid(True)
    # 添加图例
    if sat_nums <= 10:
        ax.legend(loc='upper right')

    plt.show()


def get_rotation_matrix(phi, lam):
    return np.array([
        [-np.sin(lam), np.cos(lam), 0],
        [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
        [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)]
    ])