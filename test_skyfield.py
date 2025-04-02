from datetime import datetime, timezone, timedelta

from utils.TLE import request_tle_file, save_tle_file, parse_tle_data
from skyfield.api import load, wgs84
from skyfield.framelib import itrs, tirs,ICRS

if __name__ == "__main__":
    # res = request_tle_file()
    # file_path = save_tle_file(res)
    file_path = 'satellite_data\\starlink_tle_20250401_154559.txt'
    sats = parse_tle_data(file_path)
    ts = load.timescale()
    utc_time = ts.from_datetime(datetime(2025, 4, 1, 7, 47, 0, tzinfo=timezone.utc))
    for sat in sats:
        if sat.name == 'STARLINK-4344':
            pos = sat.at(utc_time)
            lat, lon = wgs84.latlon_of(pos)
            print(lat.degrees, lon.degrees)
