from datetime import datetime, timezone
from skyfield.api import load, wgs84
from skyfield.framelib import itrs, tirs, ICRS
from utils.TLE import request_tle_file, save_tle_file, parse_tle_data, filter_epoch
from utils.Plot import sky_plot
from utils.Satellites_Params import ecef_pos_vec

if __name__ == "__main__":
    # response = request_tle_file()
    # file_path = save_tle_file(response)
    ts = load.timescale()
    file_path = 'satellite_data\\starlink_tle_20250113_225155.txt'
    sats = parse_tle_data(file_path)
    start_time_utc = datetime(2025, 1, 13, 15, 10, 0, tzinfo=timezone.utc)
    pos_sats = [sat for sat in sats if sat.name == 'STARLINK-4344']
    sat = pos_sats[0]
    geocentric = sat.at(ts.from_datetime(start_time_utc))
    lat, lon = wgs84.latlon_of(geocentric)
    h = wgs84.height_of(geocentric)
    print(f'lat={lat} lon={lon} height={h.km}')
    # x, y, z, vx, vy, vz = ecef_pos_vec(sat=pos_sats[0], utc_time=start_time_utc)
    # print(f'ECEF坐标系下位置 [{x:.2f} km, {y:.2f} km, {z:.2f} km]')
    # print(f'速度信息 [{vx:.2f} km/s, {vy:.2f} km/s, {vz:.2f} km/s]')
