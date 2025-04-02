import numpy as np
from datetime import datetime, timedelta, timezone
from skyfield.api import load, wgs84
from skyfield.framelib import itrs
import matplotlib.pyplot as plt

satellites = load.tle_file('satellite_data\\starlink_tle_20241224_202700.txt')
print('Loaded', len(satellites), 'satellites')

num_satellites = [4188, 5458, 5617, 5662, 6243]

# 加载时间和卫星数据
ts = load.timescale()
start_time_utc = datetime(2024, 12, 24, 21, 0, 0, tzinfo=timezone.utc)
time_list = [ts.from_datetime(start_time_utc + timedelta(seconds=s)) for s in range(600)]

# 创建存储 ECEF 坐标的数组
sat_ecef = np.zeros((600, 5, 3))

# 地球参数
a = 6378.137  # 地球长半轴（赤道半径，单位：km）
e = 0.0818191908426  # 地球偏心率


# 定义函数，将地理坐标转换为 ECEF 坐标
def geodetic_to_ecef(lat, lon, h):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    N = a / np.sqrt(1 - e ** 2 * np.sin(lat_rad) ** 2)
    x = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
    z = ((1 - e ** 2) * N + h) * np.sin(lat_rad)
    return x, y, z


# 遍历所有时间和卫星，计算 ECEF 坐标
for i, t in enumerate(time_list):
    for j in range(len(num_satellites)):
        sat = satellites[num_satellites[j]]
        geocentric = sat.at(t)

        # 获取纬度、经度、高度
        pos = geocentric.frame_xyz(itrs)
        x, y, z = pos.m
        # 转换为 ECEF 坐标

        # print(f"ECEF 坐标: x={x:.2f} m, y={y:.2f} m, z={z:.2f} m")
        # 存储结果
        sat_ecef[i, j, 0] = x
        sat_ecef[i, j, 1] = y
        sat_ecef[i, j, 2] = z

phi = np.deg2rad(28.1049)  # 观测者纬度（示例：30°）
lam = np.deg2rad(112.5710)  # 观测者经度（示例：120°）
Xr, Yr, Zr = geodetic_to_ecef(28.1049, 112.5710, 0)  # 观测者ECEF（示例；实际应使用真实坐标）

print(Xr, Yr, Zr)


def get_rotation_matrix(phi, lam):
    return np.array([
        [-np.sin(lam), np.cos(lam), 0],
        [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
        [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)]
    ])


R = get_rotation_matrix(phi, lam)

az_by_sv = []  # list of arrays，az_by_sv[s] 存储第 s 颗卫星所有时刻的方位角
el_by_sv = []  # list of arrays

for s in range(5):
    az_list = []
    el_list = []
    for t in range(600):
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
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)

# 设置极坐标的方向和偏移，让 0° 对齐北方（可根据喜好修改）
ax.set_theta_direction(-1)  # 方位角逆时针增加
ax.set_theta_offset(np.pi / 2.0)  # 0°在正上方

# 一次为每颗卫星绘制
colors = ['red', 'blue', 'green', 'orange', 'yellow']  # 这里简单定义4种颜色，可根据需要扩展

for s in range(5):
    # 将仰角从弧度 -> 度，计算 zenith angle = 90 - elevation(deg)
    az_deg = np.degrees(az_by_sv[s])
    el_deg = np.degrees(el_by_sv[s])
    zen_deg = 90.0 - el_deg
    # 转成绘图用的单位
    az_rad = np.radians(az_deg)
    zen_rad = np.radians(zen_deg)

    ax.plot(
        az_rad,
        zen_rad,
        color=colors[s],
        label=f"SV {s + 1}"
    )

# 设置表面参数
ax.set_rmax(np.radians(90))
ax.set_rticks([np.radians(i) for i in [20, 40, 60, 80]])
ax.set_rlabel_position(180)
ax.set_title("Skyplot", va='bottom')

# 添加图例
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

plt.show()
