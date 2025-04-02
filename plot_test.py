from datetime import datetime, timezone
from skyfield.iokit import parse_tle_file
from skyfield.api import load
from utils.Plot import sky_plot

tle_file_path = 'satellite_data\\starlink_tle_20241224_202700.txt'
ts = load.timescale()
with load.open(tle_file_path) as f:
    satellites = list(parse_tle_file(f, ts))
print('Loaded', len(satellites), 'satellites')

num_satellites = [4188, 5458, 5617, 5662, 6243]
select_sats = [satellites[index] for index in num_satellites]

# 加载时间和卫星数据

start_time_utc = datetime(2024, 12, 24, 20, 59, 0, tzinfo=timezone.utc)

sky_plot(start_time_utc, time_list_len=120, satellites=select_sats,lat=28.1049,lon=112.5710)
