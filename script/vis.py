import math
from skyfield.api import wgs84, load
from datetime import datetime, timedelta, timezone


def is_satellite_visible_from_ground(
        ground_lat_deg,
        ground_lon_deg,
        sat_lat_deg,
        sat_lon_deg,
        sat_alt_km=500.0,
        min_elevation_deg=0.0
):
    # 1. 加载时间尺度并获取当前时间（也可换成具体 UTC 时间，例如 ts.utc(2024, 12, 24, 12, 0, 0)）
    ts = load.timescale()
    t = ts.now()

    # 2. 根据地面站经纬度创建一个“地面坐标”
    #    wgs84.latlon() 中的参数 altitude 单位为米，需要注意转换
    ground = wgs84.latlon(ground_lat_deg, ground_lon_deg, elevation_m=0.0)

    # 3. 创建“卫星坐标”
    #    假设卫星此时“正好”位于 sat_lat_deg, sat_lon_deg 上空，海拔 sat_alt_km。
    #    （实际中通常需要基于轨道数据 + 观测时刻计算其真实位置）
    satellite = wgs84.latlon(
        sat_lat_deg,
        sat_lon_deg,
        elevation_m=sat_alt_km * 1000.0  # 从 km 转成 m
    )

    # 4. 用 Skyfield 计算“卫星 - 地面站”的相对位置（Difference）
    #    difference = satellite - ground 的含义：
    #      在 Skyfield 里，一个位置减去另一个位置得到“从后者看前者”的矢量位置
    difference = satellite - ground

    # 5. 计算地面站视角下的高度角 (alt)、方位角 (az) 和距离 (distance)
    topocentric = difference.at(t)
    alt, az, distance = topocentric.altaz()

    # alt/az 的单位是 Angle 类型，可用 .degrees 获取数值（单位度）
    elevation_deg = alt.degrees

    # 6. 根据最小仰角判断是否可见
    return elevation_deg >= min_elevation_deg


if __name__ == '__main__':
    ground_lat = 28.1049
    ground_lon = 112.5710
    # 卫星示例：假设在(40°N, 117°E)上空，海拔500km

    satellites = load.tle_file('satellite_data\\starlink_tle_20241224_202700.txt')
    print('Loaded', len(satellites), 'satellites')

    ts = load.timescale()
    # 3. 定义基准时间，假设是 2024-01-01 00:00:00 UTC
    start_time_utc = datetime(2024, 12, 24, 21, 0, 0, tzinfo=timezone.utc)
    count = 0
    for index, sat in enumerate(satellites):
        # 希望卫星仰角大于 10° 才认为可见
        geocentric = sat.at(ts.from_datetime(start_time_utc))  # 获取卫星在 t 时刻的地心矢量
        subpoint = geocentric.subpoint()

        sat_lon = subpoint.longitude.degrees
        sat_lat = subpoint.latitude.degrees
        sat_alt = subpoint.elevation.km
        visible = is_satellite_visible_from_ground(
            ground_lat, ground_lon,
            sat_lat, sat_lon,
            sat_alt_km=sat_alt,
            min_elevation_deg=5.0
        )

        if visible:
            count += 1

            print(index)
            # print("卫星可覆盖地面站。 (仰角 >= 40°)")
    print("共", count, "可覆盖")
