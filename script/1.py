from skyfield.api import load, wgs84
from skyfield.iokit import parse_tle_file


ts = load.timescale()
tle_file_path = 'satellite_data\\starlink_tle_20241225_174032.txt'

with load.open(tle_file_path) as f:
    sats = list(parse_tle_file(f, ts))

print("loaded", len(sats), "satellites")

# print(sats)
import numpy as np
from skyfield.framelib import itrs,tirs,ICRS
def ecef_to_lla(x_ecef):
    """
    ECEF到LLA的转换
    """
    x, y, z = x_ecef

    # WGS84椭球体参数
    a = 6378137.0  # 长半轴
    f = 1 / 298.257223563  # 扁率
    b = a * (1 - f)  # 短半轴
    e2 = 1 - (b ** 2) / (a ** 2)  # 第一偏心率平方

    # 计算经度
    lon = np.arctan2(y, x)

    # 计算纬度和高度的迭代初值
    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1 - e2))

    # 迭代计算纬度和高度
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))

    return np.array([
        np.degrees(lat),  # 纬度
        np.degrees(lon),  # 经度
        h  # 高度
    ])


# for sat in sats:
#     if sat.name == "STARLINK-4037":
#         t = ts.utc(2024, 12, 25, 18, 5, 0)
#         geocentric = sat.at(t)
#         print(geocentric.position.km)
#         lat, lon = wgs84.latlon_of(geocentric)
#         height = wgs84.height_of(geocentric)
#         print(f'{lat.degrees},{lon.degrees},{height.km}')

        # print(f'{geocentric.frame_xyz_and_velocity()}')
# print(f'{geocentric.velocity.m_per_s}')
# print(f'{geocentric.position.km}')
# lat, lon = wgs84.latlon_of(geocentric)
# print(f'lat {lat}, lon {lon}')
