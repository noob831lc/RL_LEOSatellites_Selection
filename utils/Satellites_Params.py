from typing import Annotated

import numpy as np
from datetime import datetime
from skyfield.sgp4lib import EarthSatellite
from skyfield.api import load
from skyfield.framelib import itrs, tirs
from numpy.typing import NDArray

Vector3 = Annotated[NDArray[np.float64], 3]
Matrix_Nx3 = Annotated[NDArray[np.float64], (..., 3)]

# 有关于卫星的参数的分析都在这里

ts = load.timescale()


# 卫星对地面的可见性分析
def is_satellite_visible_from_ground(
        ground_lat_deg,
        ground_lon_deg,
        sat_lat_deg,
        sat_lon_deg,
        sat_alt_km,
        min_elevation_deg=45
):
    pass


# 卫星链路功耗预算计算
def link_budget():
    pass


# 输入单个卫星对象,根据其所在的UTC时间,得到ECEF参考坐标系下的位置与速度
def ecef_pos_vec(sat: EarthSatellite, utc_time: datetime):
    geocentric = sat.at(ts.from_datetime(utc_time))
    distance, velocity = geocentric.frame_xyz_and_velocity(tirs)
    x, y, z = distance.km
    vx, vy, vz = velocity.km_per_s
    return x, y, z, vx, vy, vz


# 卫星GDOP计算
def calculate_gdop(receiver_position: Vector3, satellite_positions: Matrix_Nx3) -> float:
    """
    计算GDOP（几何精度因子）。

    参数：
        receiver_position (numpy array): 接收机在ECEF坐标系下的位置，形状为 (3,)。
        satellite_positions (numpy array): 卫星在ECEF坐标系下的位置，形状为 (n, 3)。

    返回：
        gdop (float): 几何精度因子。
    """
    num_satellites = satellite_positions.shape[0]
    if num_satellites < 4:
        raise ValueError("至少需要4颗卫星计算GDOP")

    # 构建设计矩阵 H
    H = []
    for sat_pos in satellite_positions:
        diff = sat_pos - receiver_position
        distance = np.linalg.norm(diff)
        if distance == 0:
            raise ValueError("卫星和接收机位置重合，距离不能为零")
        H.append(np.append(diff / distance, 1))  # 添加归一化方向向量和1
    H = np.array(H)

    # 计算 (H^T H)^(-1)
    HT = H.T
    HTH_inv = np.linalg.inv(HT @ H)

    # 提取对角线元素并计算GDOP
    gdop = np.sqrt(np.trace(HTH_inv))
    return gdop
