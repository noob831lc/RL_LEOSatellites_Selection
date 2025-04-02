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

# 设定观测站位置
OBSERVER_LAT = 45.0  # 北纬15度
OBSERVER_LON = -80.0  # 西经80度
OBSERVER_ELEVATION = 100  # 海拔100米
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


def plot_visibility_bar(times, visible_counts):
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
    plt.title(f'Starlink卫星可见数量 (观测站: {OBSERVER_LAT}°N, {OBSERVER_LON}°W, {OBSERVER_ELEVATION}m)', fontsize=14)
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


def main():
    # 获取TLE文件路径
    tle_path = 'satellite_data\\starlink_tle_20241224_202700.txt'

    # 加载卫星数据
    ts = load.timescale()
    with load.open(tle_path) as f:
        satellites = list(parse_tle_file(f, ts))
    print('Loaded', len(satellites), 'satellites')

    # 创建观测站对象
    observer = Topos(latitude_degrees=OBSERVER_LAT,
                     longitude_degrees=OBSERVER_LON,
                     elevation_m=OBSERVER_ELEVATION)

    # 设置分析时间范围（当前时间开始的24小时）
    time_start = start_time_utc = datetime(2024, 12, 23, 1, 30, 0, tzinfo=timezone.utc)
    time_end = time_start + timedelta(hours=20)

    print(f"分析时间范围: {time_start} 到 {time_end}")

    # 分析可见性
    times, visible_counts = analyze_visibility(satellites, observer, time_start, time_end)

    # 输出统计信息
    max_visible = max(visible_counts)
    avg_visible = sum(visible_counts) / len(visible_counts)

    print(f"分析结果:")
    print(f"平均可见卫星数: {avg_visible:.2f}")
    print(f"最大可见卫星数: {max_visible}")

    # 绘制图表
    plot_visibility_bar(times, visible_counts)
    print("可见性分析图表已保存为 'starlink_visibility.png'")

    # 输出每个时间点的可见卫星数
    print("\n每个时间点的可见卫星数:")
    for i, (t, count) in enumerate(zip(times, visible_counts)):
        print(f"{t.strftime('%Y-%m-%d %H:%M:%S')}: {count} 颗卫星可见")
        # 只显示前10个时间点，避免输出过多
        if i >= 9 and len(times) > 10:
            print("...")
            break


if __name__ == "__main__":
    main()
