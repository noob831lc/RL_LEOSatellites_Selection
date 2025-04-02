from datetime import datetime, timezone, timedelta

import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, wgs84
from utils.TLE import request_tle_file, save_tle_file, parse_tle_data, filter_epoch

file_path = 'satellite_data\\starlink_tle_20250106_144506.txt'
sats = parse_tle_data(file_path)
ELEVATION_MASK = 15.0  # 截止高度角(度)

start_time_utc = datetime(2025, 1, 6, 2, 0, 0, tzinfo=timezone.utc)
end_time_utc = datetime(2025, 1, 6, 6, 0, 0, tzinfo=timezone.utc)
time_step_minutes = 30  # 时间步长(分钟)

selected_sats = filter_epoch(sats, start_time_utc)
# 不同纬度观测站(示例:经度固定40°E, 海拔100m, 只变动纬度)
latitudes = [45, 30, 15, 0, -15, -30, -45]
longitude = 40.0
alt_m = 100.0

# ========== 1. 加载卫星 ==========
ts = load.timescale()

# ========== 2. 时间序列 ==========
current_time = start_time_utc
time_list = []
while current_time <= end_time_utc:
    time_list.append(ts.from_datetime(current_time))
    current_time += timedelta(minutes=time_step_minutes)

avg_visible_list = []  # 记录每个纬度的"日均可见星数"

for lat in latitudes:
    # 构建测站
    station = wgs84.latlon(lat, longitude, alt_m)

    # 每个时刻可见数量
    visible_counts_each_epoch = []

    for t in time_list:
        count_visible = 0
        for sat_obj in selected_sats:
            difference = sat_obj - station
            topocentric = difference.at(t)
            alt_deg, az_deg, distance = topocentric.altaz()
            if alt_deg.degrees >= ELEVATION_MASK:
                count_visible += 1
        visible_counts_each_epoch.append(count_visible)

    # 计算该纬度的一日平均可见卫星数
    mean_visible = np.mean(visible_counts_each_epoch)
    avg_visible_list.append(mean_visible)

# ========== 4. 绘制柱状图 ==========
plt.figure(figsize=(8, 5), dpi=100)
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
# 让 bar 的顺序从左到右与 latitudes 对应
# 为方便比较，可以从北纬45度到南纬45度，这里 latitudes 本身是由大到小 (-45在末尾)
# 若需要从 -45 到 45 递增，只需 latitudes 倒序或者改数组即可
x_positions = np.arange(len(latitudes))  # [0,1,2...]

plt.bar(x_positions, avg_visible_list, color='skyblue', width=0.6, edgecolor='blue')

# 设置 x 轴刻度和标签
plt.xticks(x_positions, [f"{lat}°" for lat in latitudes])
plt.xlabel("观测站纬度")
plt.ylabel("日均可见卫星数")
plt.title("不同纬度观测站的星座可见卫星数（示例）")

# 在柱状图顶端显示数值
for i, v in enumerate(avg_visible_list):
    plt.text(i, v + 0.5, f"{v:.1f}", ha='center', va='bottom', color='black')

plt.tight_layout()
plt.show()