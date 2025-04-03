from skyfield.api import wgs84, load
from skyfield.toposlib import GeographicPosition
from skyfield.sgp4lib import EarthSatellite
from datetime import datetime, timezone
from skyfield.framelib import itrs, tirs, ICRS

from TLE import timescale as ts

alt = 15  # 仰角阈值（单位：度）

def observer(
    latitude_degrees: float = 0.0, 
    longitude_degrees: float = 0.0, 
    elevation_m: float = 0.0
) -> GeographicPosition:
    """
    Create an observer at the given latitude and longitude.

    Returns:
        wgs84.GeographicPosition: An observer object.
    """
    return wgs84.latlon(latitude_degrees, longitude_degrees, elevation_m)


def ecef_pos_and_velocity(
    satellite: EarthSatellite, 
    utc_time: datetime
) -> tuple:
    """
    Create an ECEF position.

    Returns:
        An ECEF position and velocity in meters
    """
    t = ts.from_datetime(utc_time)
    distance, velocity = satellite.at(t).frame_xyz_and_velocity(tirs)
    x, y, z = distance.km
    vx, vy, vz = velocity.km_per_s
    return x, y, z, vx, vy, vz


def visable(
    sat: EarthSatellite, 
    observer: GeographicPosition, 
    utc: datetime
) -> bool:
    """
    判断 Satellite 对象是否可见（通过计算仰角 > xx°）
    """
    t = ts.from_datetime(utc)
    difference = sat.sat_object - observer
    topocentric = difference.at(t)
    alt, az, _ = topocentric.altaz()
    return alt.degrees > alt





