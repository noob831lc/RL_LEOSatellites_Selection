from skyfield.api import load, wgs84
from skyfield.iokit import parse_tle_file

from vis import is_satellite_visible_from_ground

ts = load.timescale()
tle_file_path = 'satellite_data\\starlink_tle_20241224_202700.txt'

with load.open(tle_file_path) as f:
    sats = list(parse_tle_file(f, ts))

print("loaded", len(sats), "satellites")
# print(sats)

t = ts.utc(2024, 12, 24, 20, 30, 0)
# position = sats[1].at(t)
# print(position.frame_xyz(itrs).km)
# p = wgs84.geographic_position_of()
# print(f'{p.latitude.degrees},{p.longitude.degrees},{p.elevation.km}')
# lat, lon = wgs84.latlon_of(position)
# height = wgs84.height_of(position)
# print(f'{lat.degrees},{lon.degrees},{height.km}')
# print(f'{geocentric.frame_xyz_and_velocity()}')
# print(f'{geocentric.velocity.m_per_s}')
# print(f'{geocentric.position.km}')
# lat, lon = wgs84.latlon_of(geocentric)
# print(f'lat {lat}, lon {lon}')


ground_lat = 28.1049
ground_lon = 112.5710
count = 0
for sat in sats:
    # 希望卫星仰角大于 10° 才认为可见
    position = sat.at(t)
    p = wgs84.geographic_position_of(position)
    sat_lon = p.longitude.degrees
    sat_lat = p.latitude.degrees
    sat_alt = p.elevation.km
    visible = is_satellite_visible_from_ground(
        ground_lat, ground_lon,
        sat_lat, sat_lon,
        sat_alt_km=sat_alt,
        min_elevation_deg=4.0
    )

    if visible:
        count += 1
        print(sat.name, "卫星可覆盖地面站。 (仰角 >= 40°)")
print("共", count, "可覆盖")
