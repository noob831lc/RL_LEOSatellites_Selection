from datetime import datetime, timezone, timedelta
from skyfield.api import load
from skyfield.framelib import tirs
from skyfield.toposlib import wgs84

from sat_selection import ossa_selection, update_satellite_info, Satellite, fcsdp_selection, maxele_selection
from utils.tle import request_tle_file, save_tle_file, parse_tle_data, filter_epoch
from utils.sky_plot import sky_plot


def visable(sat, observer, start_time_utc):
    difference = sat - observer
    time_list = [ts.from_datetime(start_time_utc + timedelta(seconds=s)) for s in range(120)]
    for t in time_list:
        topocentric = difference.at(t)
        alt, az, distance = topocentric.altaz()
        if alt.degrees < 25:
            return False
    return True


if __name__ == "__main__":
    # response = request_tle_file()
    # file_path = save_tle_file(response)

    file_path = 'satellite_data\\starlink_tle_20250106_144506.txt'
    sats = parse_tle_data(file_path)
    start_time_utc1 = datetime(2025, 1, 6, 14, 0, 0, tzinfo=timezone.utc)
    start_time_utc2 = datetime(2025, 1, 6, 14, 2, 0, tzinfo=timezone.utc)
    start_time_utc3 = datetime(2025, 1, 6, 14, 4, 0, tzinfo=timezone.utc)
    start_time_utc4 = datetime(2025, 1, 6, 14, 6, 0, tzinfo=timezone.utc)
    ts = load.timescale()

    # 定义观测者（用户）位置（可采用文中示例：[126.68°E, 45.75°N, 100 m]）
    observer = wgs84.latlon(45.75, 126.68, elevation_m=100)
    pos_sats1 = [sat for sat in sats if visable(sat, observer, start_time_utc1)]
    pos_sats2 = [sat for sat in sats if visable(sat, observer, start_time_utc2)]
    pos_sats3 = [sat for sat in sats if visable(sat, observer, start_time_utc3)]
    pos_sats4 = [sat for sat in sats if visable(sat, observer, start_time_utc4)]
    print(len(pos_sats1), len(pos_sats2), len(pos_sats3), len(pos_sats4))
    sky_plot(start_time_utc1, time_list_len=120, satellites=pos_sats1, lat=45.75, lon=126.68)
    sky_plot(start_time_utc2, time_list_len=120, satellites=pos_sats2, lat=45.75, lon=126.68)
    sky_plot(start_time_utc3, time_list_len=120, satellites=pos_sats3, lat=45.75, lon=126.68)
    sky_plot(start_time_utc4, time_list_len=120, satellites=pos_sats4, lat=45.75, lon=126.68)
    #
    user_itrs = observer.at(ts.from_datetime(start_time_utc1)).position.km
    osat = [Satellite(sat) for sat in pos_sats1]
    for sat in osat:
        update_satellite_info(sat, observer, ts.from_datetime(start_time_utc1))
    combo1, dop = maxele_selection(osat, user_itrs, 4)
    combo2, dop = ossa_selection(osat, user_itrs, 4)
    combo3, dop = fcsdp_selection(osat, user_itrs, 4)
    print(combo1, dop)
    print(combo2, dop)
    print(combo3, dop)
    combo_sat1 = [sat.sat_object for sat in combo1]
    combo_sat2 = [sat.sat_object for sat in combo2]
    combo_sat3 = [sat.sat_object for sat in combo3]
    sky_plot(start_time_utc1, time_list_len=120, satellites=combo_sat1, lat=45.75, lon=126.68)
    sky_plot(start_time_utc1, time_list_len=120, satellites=combo_sat2, lat=45.75, lon=126.68)
    sky_plot(start_time_utc1, time_list_len=120, satellites=combo_sat3, lat=45.75, lon=126.68)
