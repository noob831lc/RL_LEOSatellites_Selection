import numpy as np
from skyfield.api import EarthSatellite
from skyfield.toposlib import GeographicPosition
from datetime import datetime
from typing import Optional, Any
from utils.tle import timescale as ts


class Satellite:
    def __init__(self, sat_object: EarthSatellite) -> None:
        """
        sat_object: Skyfield 中解析 TLE 后返回的 EarthSatellite 对象
        """
        self.sat_object: EarthSatellite = sat_object
        self.name: str = sat_object.name
        # 卫星的 ECEF 坐标（单位 m）
        self.position: Optional[np.ndarray] = None
        # 仰角、方位角（单位：度）
        self.elevation: Optional[float] = None
        self.azimuth: Optional[float] = None
        # GEV 特征参数：phi（垂直角，单位：弧度）、beta（水平特征角，单位：弧度）
        self.phi: Optional[float] = None
        self.beta: Optional[float] = None
        # 权重（用于排序备用）
        self.weight: Optional[float] = None

    def __repr__(self) -> str:
        return f"{self.name}"
    

def update_satellite_info(
    sat: Satellite, 
    observer: GeographicPosition, 
    utc: datetime
) -> None:
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