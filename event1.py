from skyfield.api import load, Topos, EarthSatellite
from skyfield.timelib import Time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
from datetime import datetime, timedelta, timezone
from skyfield.iokit import parse_tle_file

from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

# 设定观测参数
MIN_ELEVATION = 25.0  # 最低高度角25度


def analyze_visibility(satellites, observer, time_start, time_end, step_minutes=10):
    """分析给定时间段内可见卫星数量"""
    times = []
    visible_counts = []

    current_time = time_start
    while current_time <= time_end:
        # 转换为Skyfield时间对象
        ts = load.timescale()
        t = ts.from_datetime(current_time)

        # 计算每颗卫星的位置
        visible_count = 0
        for sat in satellites:
            try:
                # 计算卫星相对于观测者的位置
                difference = sat - observer
                topocentric = difference.at(t)

                # 获取高度角
                alt, az, distance = topocentric.altaz()

                # 如果高度角大于最低要求，则认为可见
                if alt.degrees > MIN_ELEVATION:
                    visible_count += 1
            except Exception as e:
                # 忽略计算错误的卫星
                pass

        times.append(current_time)
        visible_counts.append(visible_count)
        current_time += timedelta(minutes=step_minutes)

    return times, visible_counts


def plot_visibility_bar(times, visible_counts, observer_info):
    """绘制可见卫星数量随时间变化的柱状图"""
    plt.figure(figsize=(15, 7))

    # 转换时间为matplotlib日期格式
    times_mpl = mdates.date2num(times)

    # 根据时间点数量调整柱宽
    bar_width = min(0.01, 0.8 * (times_mpl[1] - times_mpl[0])) if len(times_mpl) > 1 else 0.01

    # 绘制柱状图
    plt.bar(times_mpl, visible_counts, width=bar_width, color='skyblue', edgecolor='navy')

    # 设置x轴格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))

    # 添加标题和标签
    lat, lon, elev = observer_info
    lat_label = f"{abs(lat)}°{'N' if lat >= 0 else 'S'}"
    lon_label = f"{abs(lon)}°{'E' if lon >= 0 else 'W'}"
    plt.title(f'Starlink卫星可见数量 (观测站: {lat_label}, {lon_label}, {elev}m)', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('可见卫星数量', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # 调整x轴标签
    plt.xticks(rotation=45)

    # 添加水平辅助线
    if visible_counts:
        avg_visible = sum(visible_counts) / len(visible_counts)
        plt.axhline(y=avg_visible, color='red', linestyle='-', alpha=0.5, label=f'平均值: {avg_visible:.2f}')
        plt.legend()

    plt.tight_layout()

    # 保存图表
    plt.savefig('starlink_visibility_bar.png', dpi=300)
    plt.close()


def analyze_multiple_stations():
    """使用NumPy优化的多观测站分析函数"""
    latitudes = [45, 30, 15, 0, -15, -30, -45]  # 北纬45°到南纬45°
    longitudes = [-80, 40, 160]  # 80°W, 40°E, 160°E
    elevation = 100  # 海拔高度100米
    longitude_labels = ["80°W", "40°E", "160°E"]

    # 加载卫星数据
    tle_path = 'satellite_data\\starlink_tle_20241224_202700.txt'
    ts = load.timescale()
    with load.open(tle_path) as f:
        satellites = list(parse_tle_file(f, ts))
    print('Loaded', len(satellites), 'satellites')

    # 设置分析时间范围
    time_start = datetime(2024, 12, 23, 1, 30, 0, tzinfo=timezone.utc)
    time_end = time_start + timedelta(hours=24)
    step_minutes = 30

    # 创建时间点数组
    time_steps = []
    current_time = time_start
    while current_time <= time_end:
        time_steps.append(current_time)
        current_time += timedelta(minutes=step_minutes)

    # 转换为Skyfield时间对象数组
    ts_array = np.array([ts.from_datetime(t) for t in time_steps])

    # 预分配结果数组 - 形状为 [纬度, 经度, 时间点]
    results_array = np.zeros((len(latitudes), len(longitudes), len(time_steps)))

    # 创建所有观测站对象
    observers = []
    for lat in latitudes:
        for lon in longitudes:
            observers.append(Topos(latitude_degrees=lat, longitude_degrees=lon, elevation_m=elevation))

    # 使用NumPy的并行能力批量计算
    # 注意：Skyfield API可能不直接支持向量化，需要适配
    for time_idx, t in enumerate(ts_array):
        # 批量计算每个观测站的可见卫星
        for obs_idx, observer in enumerate(observers):
            lat_idx = obs_idx // len(longitudes)
            lon_idx = obs_idx % len(longitudes)

            # 计算可见卫星数量
            visible_count = 0
            for sat in satellites:
                try:
                    difference = sat - observer
                    topocentric = difference.at(t)
                    alt, az, distance = topocentric.altaz()
                    if alt.degrees > MIN_ELEVATION:
                        visible_count += 1
                except Exception:
                    pass

            results_array[lat_idx, lon_idx, time_idx] = visible_count

    # 计算每个观测站的平均和最大可见卫星数
    avg_visible = np.mean(results_array, axis=2)
    max_visible = np.max(results_array, axis=2)

    # 构建结果字典
    results = {}
    for lat_idx, lat in enumerate(latitudes):
        results[lat] = {}
        for lon_idx, lon in enumerate(longitudes):
            results[lat][lon] = {
                'avg_visible': avg_visible[lat_idx, lon_idx],
                'max_visible': max_visible[lat_idx, lon_idx]
            }
            print(f"观测站 ({lat}°, {longitudes[lon_idx]}°, {elevation}m) 平均可见卫星数: {avg_visible[lat_idx, lon_idx]:.2f}")

    # 绘制结果
    plot_visible_satellites_bar(latitudes, longitudes, longitude_labels, results)

    return results


def plot_visible_satellites_bar(latitudes, longitudes, longitude_labels, results):
    """专门绘制图1中的(b)子图 - 可见卫星数的柱状图"""
    # 创建图表
    plt.figure(figsize=(12, 6))

    # 配置柱状图位置
    bar_width = 0.25
    x = np.arange(len(latitudes))

    # 颜色设置 - 蓝色系，与参考图保持一致
    colors = ['#FFA07A', '#87CEFA', '#6CF0C0']  # 三种蓝色

    # 绘制三组柱状图（对应三个经度）
    for i, lon_idx in enumerate(range(len(longitudes))):
        lon = longitudes[lon_idx]

        # 提取可见卫星数数据
        sat_values = [results[lat][lon]['max_visible'] for lat in latitudes]

        # 计算柱子位置
        positions = x + (i - 1) * bar_width

        # 绘制柱状图
        plt.bar(positions, sat_values, bar_width, color=colors[i], label=longitude_labels[i])

    # 添加标题和标签
    plt.title('可见卫星数', fontsize=14)
    plt.xlabel('观测站纬度', fontsize=14)
    plt.ylabel('可见卫星数', fontsize=14)

    # 添加图例
    plt.legend()

    # 设置X轴刻度和标签
    plt.xticks(x, [f"{abs(lat)}°{'S' if lat < 0 else 'N' if lat > 0 else ''}" for lat in latitudes])

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    # 添加垂直虚线分隔不同纬度
    for i in range(1, len(latitudes)):
        plt.axvline(x=i - 0.5, color='k', linestyle=':', alpha=0.3)

    # 设置Y轴范围，确保与参考图一致

    # 调整布局
    plt.tight_layout()

    # 保存图表
    plt.savefig('starlink_visible_satellites.png', dpi=300)
    plt.show()


def main():
    print("==== Starlink星座卫星可视性分析 ====")

    analyze_multiple_stations()


if __name__ == "__main__":
    main()
