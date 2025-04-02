import numpy as np
from math import sin, cos, sqrt, atan2, degrees, radians, asin

# 地球参数
EARTH_RADIUS = 6371.0  # 地球半径，单位：公里


def geodetic_to_ecef(lat, lon, h):
    """
    将地理坐标（经度、纬度、高度）转换为ECEF坐标系。
    输入：
        lat: 纬度，单位：度
        lon: 经度，单位：度
        h: 高度，单位：公里
    输出：
        (x, y, z) ECEF坐标，单位：公里
    """
    # WGS84参数
    a = 6378.137  # 长半轴，单位：公里
    e_sq = 6.69437999014e-3  # 第一偏心率平方

    lat_rad = radians(lat)
    lon_rad = radians(lon)

    N = a / sqrt(1 - e_sq * sin(lat_rad) ** 2)

    x = (N + h) * cos(lat_rad) * cos(lon_rad)
    y = (N + h) * cos(lat_rad) * sin(lon_rad)
    z = ((1 - e_sq) * N + h) * sin(lat_rad)

    return np.array([x, y, z])


def calculate_elevation_angle(gs_ecef, sat_ecef):
    """
    计算卫星相对于地面站的仰角。
    输入：
        gs_ecef: 地面站ECEF坐标，单位：公里
        sat_ecef: 卫星ECEF坐标，单位：公里
    输出：
        仰角，单位：度
    """
    # 计算地面站到卫星的向量
    vector = sat_ecef - gs_ecef

    # 计算距离
    distance = np.linalg.norm(vector)

    # 单位向量
    vector_unit = vector / distance

    # 地面站的法向量（地心指向地面站）
    gs_norm = gs_ecef / np.linalg.norm(gs_ecef)

    # 计算仰角
    elevation_rad = asin(np.dot(vector_unit, gs_norm))
    elevation_deg = degrees(elevation_rad)

    return elevation_deg, distance


def calculate_path_loss(d_km, frequency_ghz):
    """
    计算自由空间路径损耗（FSPL）。
    输入：
        d_km: 距离，单位：公里
        frequency_ghz: 频率，单位：GHz
    输出：
        路径损耗，单位：dB
    """
    # FSPL公式： Lp(dB) = 20*log10(d) + 20*log10(f) + 92.45
    Lp = 20 * np.log10(d_km) + 20 * np.log10(frequency_ghz) + 92.45
    return Lp


def link_budget(Pt_dbm, Gt_dbi, Gr_dbi, Lp_db, Ls_db):
    """
    计算接收功率。
    输入：
        Pt_dbm: 发射功率，单位：dBm
        Gt_dbi: 发射天线增益，单位：dBi
        Gr_dbi: 接收天线增益，单位：dBi
        Lp_db: 路径损耗，单位：dB
        Ls_db: 系统损耗，单位：dB
    输出：
        接收功率，单位：dBm
    """
    Pr_dbm = Pt_dbm + Gt_dbi + Gr_dbi - Lp_db - Ls_db
    return Pr_dbm


def main():
    # 示例数据

    # 地面站位置（经度、纬度、海拔高度）
    gs_lon = 0.0  # 经度，单位：度
    gs_lat = 0.0  # 纬度，单位：度
    gs_alt = 0.0  # 海拔高度，单位：公里

    # 卫星位置（经度、纬度、轨道高度）
    # 例如，赤道上空500公里高度
    sat_lat = 0.0  # 纬度，单位：度
    sat_lon = 10.0  # 经度，单位：度
    sat_alt = EARTH_RADIUS + 500  # 轨道高度，单位：公里

    # 链路预算参数
    frequency = 2.0  # 频率，单位：GHz
    Pt_watts = 10  # 发射功率，单位：瓦特
    Pt_dbm = 10 * np.log10(Pt_watts * 1000)  # 转换为dBm
    Gt_dbi = 30  # 发射天线增益，单位：dBi
    Gr_dbi = 30  # 接收天线增益，单位：dBi
    Ls_db = 2  # 系统损耗，单位：dB
    Pr_sensitivity = -90  # 接收灵敏度，单位：dBm
    min_elevation = 10.0  # 最小仰角阈值，单位：度

    # 转换坐标
    gs_ecef = geodetic_to_ecef(gs_lat, gs_lon, gs_alt)
    sat_ecef = geodetic_to_ecef(sat_lat, sat_lon, sat_alt)

    # 计算仰角和距离
    elevation, distance = calculate_elevation_angle(gs_ecef, sat_ecef)

    print(f"卫星与地面站之间的距离: {distance:.2f} km")
    print(f"仰角: {elevation:.2f} 度")

    # 判断可视性
    if elevation >= min_elevation:
        print("卫星在地面站的可视范围内。")
        visible = True
    else:
        print("卫星不在地面站的可视范围内。")
        visible = False

    if visible:
        # 计算路径损耗
        Lp = calculate_path_loss(distance, frequency)
        print(f"路径损耗 (Lp): {Lp:.2f} dB")

        # 计算接收功率
        Pr = link_budget(Pt_dbm, Gt_dbi, Gr_dbi, Lp, Ls_db)
        print(f"接收功率 (Pr): {Pr:.2f} dBm")

        # 判断链路预算
        if Pr >= Pr_sensitivity:
            print("通信链路可行。")
        else:
            print("通信链路不可行，接收功率低于灵敏度。")
    else:
        print("由于卫星不可视，无法进行链路预算。")


if __name__ == "__main__":
    main()
