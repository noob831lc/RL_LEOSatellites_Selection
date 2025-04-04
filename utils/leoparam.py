from datetime import datetime, timedelta
from skyfield.api import wgs84
from skyfield.toposlib import GeographicPosition
from skyfield.sgp4lib import EarthSatellite
from skyfield.framelib import tirs

from utils.tle import timescale as ts

ELEVATION_MASK_ANGLE= 15  # 截止高度角（单位：度）

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

def ecef_pos(
    satellite: EarthSatellite, 
    utc_time: datetime
) -> tuple:
    """
    Create an ECEF position.

    Returns:
        An ECEF position in meters
    """
    t = ts.from_datetime(utc_time)
    distance = satellite.at(t).frame_xyz(tirs)
    x, y, z = distance.km
    return x, y, z


def visable(
    sat: EarthSatellite, 
    ground_station: GeographicPosition, 
    utc: datetime
) -> bool:
    """
    判断 Satellite 对象是否可见（通过计算仰角 > xx°）
    """
    t = ts.from_datetime(utc)
    difference = sat.sat_object - ground_station
    topocentric = difference.at(t)
    alt, _ , _ = topocentric.altaz()
    return alt.degrees > ELEVATION_MASK_ANGLE


def visable_period(sat, observer, start_time_utc):
    difference = sat - observer
    time_list = [ts.from_datetime(start_time_utc + timedelta(seconds=s)) for s in range(120)]
    for t in time_list:
        topocentric = difference.at(t)
        alt, az, distance = topocentric.altaz()
        if alt.degrees < 25:
            return False
    return True




