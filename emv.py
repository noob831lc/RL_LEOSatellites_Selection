import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, EarthSatellite, wgs84
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


tle_file = 'satellite_data\\starlink_tle_20250106_144506.txt'
satellites_tle = load.tle_file(tle_file)
print(f"加载了 {len(satellites_tle)} 颗卫星 TLE 数据.")
ts = load.timescale()

observer = wgs84.latlon(45.75, 126.68, elevation_m=100)

time_points = [
    ts.utc(2025, 1, 6, 14, 2, 0),
    ts.utc(2025, 1, 6, 14, 4, 0),
    ts.utc(2025, 1, 6, 14, 6, 0),
    ts.utc(2025, 1, 6, 14, 8, 0)
]

def visable(sat, observer, t):
    difference = sat - observer
    topocentric = difference.at(t)
    alt, az, distance = topocentric.altaz()
    return alt.degrees > 15

EMV_list = []
sat_pos_list = []

common_elements = None
for t in time_points:
    satellites = [sat for sat in satellites_tle if visable(sat, observer, t)]
    if common_elements is None:
        common_elements = set(satellites)
    else:
        common_elements &= set(satellites)
sat = list(common_elements)[0]

for t in time_points:
    sat_pos = sat.at(t).position.km
    sat_vel = sat.at(t).velocity.km_per_s
    user_pos = observer.at(t).position.km
    r = sat_pos - user_pos
    r_norm = np.linalg.norm(r)
    EMV = np.cross(r, np.cross(r, sat_vel)) / (r_norm ** 3)
    EMV_list.append(EMV)
    sat_pos_list.append(sat_pos)

EMV_array = np.array(EMV_list)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制原点
ax.scatter(0, 0, 0, color='black', s=100, label='Origin')

# 绘制坐标轴
axis_length = 1.0  # 坐标轴长度
ax.quiver(0, 0, 0, axis_length, 0, 0, color='gray', arrow_length_ratio=0.1, linewidth=2)
ax.quiver(0, 0, 0, 0, axis_length, 0, color='gray', arrow_length_ratio=0.1, linewidth=2)
ax.quiver(0, 0, 0, 0, 0, axis_length, color='gray', arrow_length_ratio=0.1, linewidth=2)
ax.text(axis_length * 1.05, 0, 0, "X", fontsize=14, color='gray')
ax.text(0, axis_length * 1.05, 0, "Y", fontsize=14, color='gray')
ax.text(0, 0, axis_length * 1.05, "Z", fontsize=14, color='gray')

# 为了区分不同时间点的 EMV，设置不同颜色
colors = ['red', 'blue', 'green', 'magenta']
scale_factor = 1e1
for i, EMV in enumerate(EMV_array):
    # 绘制以原点为起点的 EMV 向量
    EMV_scaled = EMV * scale_factor
    ax.quiver(0, 0, 0, EMV_scaled[0], EMV_scaled[1], EMV_scaled[2],
              color=colors[i], arrow_length_ratio=0.1, linewidth=2,
              label=f'Time {i+1}')
    # 在向量末端额外标注文本（T1, T2, ...）
    ax.text(EMV_scaled[0]*1.1, EMV_scaled[1]*1.1, EMV_scaled[2]*1.1, f'T{i+1}', fontsize=12, color=colors[i])

# 设置坐标轴标签与标题
max_range = np.max(np.abs(EMV_array * scale_factor)) * 1.2
# ax.set_xlim([-1.5*max_range, 1.5*max_range])
# ax.set_ylim([-1.5*max_range, 1.5*max_range])
# ax.set_zlim([-1.5*max_range, 1.5*max_range])
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("EMV Vectors at Four Time Points")

ax.legend()
ax.grid(True)
plt.show()