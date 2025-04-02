import numpy as np
from datetime import datetime, timedelta, timezone
from skyfield.api import load
import matplotlib.pyplot as plt
from skyfield.framelib import itrs, ICRS, tirs, ICRS_to_J2000
from skyfield.toposlib import Topos


# def sky_plot(start_time_utc,
#              time_list_len,
#              satellites,
#              lat, lon):
#     ts = load.timescale()
#     time_list = [ts.from_datetime(start_time_utc + timedelta(seconds=s)) for s in range(time_list_len)]
#     location = Topos(latitude_degrees=lat,
#                      longitude_degrees=lon,
#                      elevation_m=100)
#     az_by_sv = []  # list of arrays，az_by_sv[s] 存储第 s 颗卫星所有时刻的方位角
#     el_by_sv = []  # list of arrays
#     for sat in satellites:
#         az_list = []
#         el_list = []
#         for t in time_list:
#             difference = sat - location
#             apparent = difference.at(t)
#             alt, az, distance = apparent.altaz()
#
#             r = 90 - alt.degrees
#             theta = np.radians(az.degrees)
#
#             az_list.append(theta)
#             el_list.append(r)
#
#         az_by_sv.append(np.array(az_list))
#         el_by_sv.append(np.array(el_list))
#
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, polar=True)
#
#     # 设置极坐标系
#     ax.set_theta_zero_location('N')  # 北方为0度
#     ax.set_theta_direction(-1)  # 顺时针方向
#
#     ax.grid(True)
#
#     # 颜色列表，使用循环色彩使每条轨迹更容易区分
#     # colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan',
#     #           'magenta', 'brown', 'pink', 'olive', 'gray', 'gold']
#
#     sat_nums = len(satellites)
#     for s in range(sat_nums):
#         # 将仰角和方位角从弧度转为度
#         az_deg = np.degrees(az_by_sv[s])
#         el_deg = np.degrees(el_by_sv[s])
#
#         # 直接使用仰角(el_deg)，不需要转换为天顶角
#         # 转回弧度用于绘图
#         az_rad = az_deg
#         el_rad = np.radians(el_deg)  # 使用仰角值
#         # el_rad = el_deg
#         # 选择颜色并使用虚线样式
#         # color = colors[s % len(colors)]
#
#         # 绘制轨迹，使用虚线样式
#         ax.plot(
#             az_rad,
#             el_rad,
#             linestyle='-',
#             linewidth=1.0,
#             label=f"{satellites[s].name}"
#         )
#         ax.set_rmax(np.radians(90))
#         ax.set_thetagrids(range(0, 360, 30))
#         ax.set_rlabel_position(180)
#         ax.set_rticks([np.radians(i) for i in [0, 30, 60, 90]])
#         # 设置网格线样式
#         ax.grid(True, color='gray', linestyle='-', alpha=0.3)
#
#         # 移除外部圆的显示（即90°圆）
#         # ax.spines['polar'].set_visible(False)
#
#         # 如果卫星数量不多，添加图例
#         if sat_nums <= 10:
#             ax.legend(loc='upper right', framealpha=0.7)
#
#         plt.tight_layout()
#         plt.show()

#
#
# 绘制星下点轨迹图 参数 起始时间 观测时间长度(单位 s)  观测卫星列表 观测点经纬度
def sky_plot(start_time_utc,
             time_list_len,
             satellites,
             lat, lon):
    ts = load.timescale()
    time_list = [ts.from_datetime(start_time_utc + timedelta(seconds=s)) for s in range(time_list_len)]

    # 创建地面站点对象
    location = Topos(latitude_degrees=lat,
                     longitude_degrees=lon,
                     elevation_m=100)

    az_by_sv = []  # list of arrays，az_by_sv[s] 存储第 s 颗卫星所有时刻的方位角
    el_by_sv = []  # list of arrays
    #
    for sat in satellites:
        az_list = []
        el_list = []
        for t in time_list:
            difference = sat - location
            apparent = difference.at(t)
            alt, az, distance = apparent.altaz()

            r = 90 - alt.degrees
            theta = np.radians(az.degrees)

            az_list.append(theta)
            el_list.append(r)

        az_by_sv.append(np.array(az_list))
        el_by_sv.append(np.array(el_list))

    # ----------------------------
    # 4. Skyplot 绘制 - 修改部分
    # ----------------------------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # 设置极坐标系
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

    # 颜色列表，使用循环色彩使每条轨迹更容易区分
    # colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan',
    #           'magenta', 'brown', 'pink', 'olive', 'gray', 'gold']

    sat_nums = len(satellites)
    for s in range(sat_nums):
        # 将仰角和方位角从弧度转为度
        az_deg = np.degrees(az_by_sv[s])
        el_deg = np.degrees(el_by_sv[s])

        # 直接使用仰角(el_deg)，不需要转换为天顶角
        # 转回弧度用于绘图
        az_rad = np.radians(az_deg)
        # el_rad = np.radians(el_deg)  # 使用仰角值
        el_rad = np.radians(el_deg)
        # 选择颜色并使用虚线样式
        # color = colors[s % len(colors)]

        # 绘制轨迹，使用虚线样式
        ax.plot(
            az_rad,
            el_rad,
            linestyle='-',
            linewidth=1.0,
            label=f"{satellites[s].name}"
        )
    ax.set_rmax(np.radians(90))
    ax.set_thetagrids(range(0, 360, 30))
    ax.set_rlabel_position(180)

    # 设置网格线样式
    ax.grid(True, color='gray', linestyle='-', alpha=0.3)

    # 移除外部圆的显示（即90°圆）
    # ax.spines['polar'].set_visible(False)

    # 如果卫星数量不多，添加图例
    if sat_nums <= 10:
        ax.legend(loc='upper right', framealpha=0.7)

    plt.tight_layout()
    plt.show()


def get_rotation_matrix(phi, lam):
    return np.array([
        [-np.sin(lam), np.cos(lam), 0],
        [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
        [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)]
    ])
