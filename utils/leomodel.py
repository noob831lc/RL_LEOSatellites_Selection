import numpy as np
from skyfield.api import EarthSatellite
from skyfield.toposlib import GeographicPosition
from datetime import datetime
from utils.tle import timescale as ts


class Satellite:
    def __init__(self, sat_object):
        """
        sat_object: Skyfield 中解析 TLE 后返回的 EarthSatellite 对象
        """
        self.sat_object = sat_object
        self.name = sat_object.name
        # 卫星的 ECEF 坐标（单位 m）
        self.position = None
        # 仰角、方位角（单位：度）
        self.elevation = None
        self.azimuth = None
        # GEV 特征参数：phi（垂直角，单位：弧度）、beta（水平特征角，单位：弧度）
        self.phi = None
        self.beta = None
        # 权重（用于排序备用）
        self.weight = None

    def __repr__(self):
        return f"{self.name}"
    
def update_satellite_info(sat : EarthSatellite, observer : GeographicPosition, utc: datetime) -> None:
    """
    根据观测者位置 observer 更新卫星信息：
      - 获取卫星的 ECEF 坐标（单位：m）
      - 计算仰角、方位角（单位：度）
      - 根据仰角设置 phi：仰角>=50°时赋值 2π/3，否则赋值 π/3；beta 为方位角的弧度
    """
    t = ts.from_datetime(utc)
    pos = sat.sat_object.at(t).position
    sat.position = pos.m  # 单位 m
    difference = sat.sat_object - observer
    alt, az, _ = difference.at(t).altaz()
    sat.elevation = alt.degrees
    sat.azimuth = az.degrees
    if sat.elevation >= 50:
        sat.phi = 2 * np.pi / 3
    else:
        sat.phi = np.pi / 3
    sat.beta = np.radians(sat.azimuth)