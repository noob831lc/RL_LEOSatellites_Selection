import time
from datetime import datetime, timezone
from skyfield.api import load
from utils.TLE import request_tle_file, save_tle_file, parse_tle_data
from utils.leo_param import observer, ecef_pos_and_velocity
from utils.leo_param import timescale

if __name__ == "__main__":
    # response = request_tle_file(group='starlink', format='tle')
    # file_path = save_tle_file(response, group='starlink', format='tle')
    satellites = parse_tle_data('satellite_data\starlink_tle_20250403_140811.txt')

    ground_station = observer(45.75, 126.68, 100.0)
    sat = satellites[0]
    utc_time = datetime(2025, 4, 3, 6, 10, 0, tzinfo=timezone.utc)
    t = load.timescale().from_datetime(utc_time)
    print(f'UTC-Time: {utc_time}')
    x, y, z, vx, vy, vz = ecef_pos_and_velocity(sat, utc_time)
    print(f'卫星名称: {sat.name}')
    print(f'卫星轨道: {sat.model}')
    print(f'位置坐标: {x:.2f}km, {y:.2f}km, {z:.2f}km')
    print(f'速度坐标: {vx:.2f}km/s, {vy:.2f}km/s, {vz:.2f}km/s')
    # print(ground_station)
    difference= sat - ground_station
    print(f'{difference}')
    topocentric = difference.at(t)
    alt, az, _ = topocentric.altaz()
    print(f'{alt.degrees:.2f}° {az.degrees:.2f}°')
