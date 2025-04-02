from skyfield.api import Topos, load, EarthSatellite
from datetime import datetime, timedelta, timezone


def satellites_visible_in_next_minutes(
        satellites,
        observer_lat_deg,
        observer_lon_deg,
        horizon_deg=0.0,
        minutes_ahead=5,
        time_step_seconds=1
):
    """
    计算未来指定分钟内，可见（仰角大于指定阈值）的卫星列表。
    参数：
        - satellites:      EarthSatellite 对象列表
        - observer_lat_deg, observer_lon_deg: 地面站经纬度（单位：度）
        - horizon_deg:     可见仰角阈值（单位：度, 默认为 0，即地平线）
        - minutes_ahead:   要计算的时间范围（单位：分钟, 默认为 10）
        - time_step_seconds: 步进时间间隔（单位：秒, 默认为 30）
    返回：
        - visible_sats:    在未来指定时间内出现过仰角>horizon_deg 的卫星名称列表
    """
    # 设置地面站位置
    ground_station = Topos(latitude_degrees=observer_lat_deg,
                           longitude_degrees=observer_lon_deg)

    # 生成一个时间序列
    ts = load.timescale()
    start_time = datetime(2024, 12, 24, 20, 0, 0, tzinfo=timezone.utc)
    end_time = start_time + timedelta(minutes=minutes_ahead)

    # 将 Python datetime 转换为 skyfield 时间对象
    current_time_sky = ts.from_datetime(start_time)
    end_time_sky = ts.from_datetime(end_time)

    # 生成离散采样时刻
    total_seconds = (end_time - start_time).total_seconds()
    num_steps = int(total_seconds // time_step_seconds) + 1
    time_list = [
        current_time_sky + (i * time_step_seconds) / 86400.0  # 每日秒数 = 86400
        for i in range(num_steps)
    ]

    visible_sats = set()

    # 针对每颗卫星，检查在未来时间段内是否有时刻仰角大于给定阈值
    for sat in satellites:
        for t in time_list:
            topocentric = (sat - ground_station).at(t)
            alt, az, distance = topocentric.altaz()
            if alt.degrees > horizon_deg:
                visible_sats.add(sat.name)
                break  # 找到一次仰角大于阈值就可以退出该星的循环

    return list(visible_sats)


def main():
    # 假设本地有一个 TLE 文件 'tle.txt'
    sats = load.tle_file('satellite_data\\starlink_tle_20241224_202700.txt')

    # 假设地面站位于北京（仅为示例）
    observer_lat = 28.1049
    observer_lon = 112.5710

    # 计算未来 10 分钟内可见的卫星
    visible_sats = satellites_visible_in_next_minutes(
        sats,
        observer_lat_deg=observer_lat,
        observer_lon_deg=observer_lon,
        horizon_deg=10,  # 定义地平线以上即可，或者可设置为 10 度
        minutes_ahead=5,  # 未来 10 分钟
        time_step_seconds=1
    )

    print("未来 5 分钟内可见的卫星：")
    print("数量：", len(visible_sats))
    for sat_name in visible_sats:
        print(" -", sat_name)


if __name__ == '__main__':
    main()
