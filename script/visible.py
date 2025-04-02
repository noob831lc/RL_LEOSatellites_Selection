import math

# 假设地球平均半径（单位：km，可按需使用更精确的椭球模型）
EARTH_RADIUS = 6371.0


def lat_lon_alt_to_ecef(lat_deg, lon_deg, alt_km=0.0):
    """
    将给定的地理坐标（纬度、经度、海拔）转换为地心坐标系(ECEF)下的 x, y, z。
    参数:
        lat_deg (float): 纬度，单位度
        lon_deg (float): 经度，单位度
        alt_km  (float): 高度，单位km（相对于地球表面）
    返回:
        (x, y, z) (float, float, float)
    """
    # 转为弧度
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    # 计算在 ECEF 下的坐标
    # 假设地球半径为 EARTH_RADIUS, 卫星海拔为 alt_km
    r = EARTH_RADIUS + alt_km
    x = r * math.cos(lat_rad) * math.cos(lon_rad)
    y = r * math.cos(lat_rad) * math.sin(lon_rad)
    z = r * math.sin(lat_rad)
    return x, y, z


def is_satellite_visible_from_ground(
        ground_lat_deg, ground_lon_deg,
        sat_lat_deg, sat_lon_deg, sat_alt_km=500.0
):
    """
    判断给定卫星位置(纬度、经度、高度)是否覆盖到地面站(纬度、经度)。
    这里采用简单的0°仰角判定（即只要卫星在地平线以上，就视为可见）。

    参数:
        ground_lat_deg (float): 地面站纬度（度）
        ground_lon_deg (float): 地面站经度（度）
        sat_lat_deg    (float): 卫星所在纬度（度）
        sat_lon_deg    (float): 卫星所在经度（度）
        sat_alt_km     (float): 卫星距地面的高度（km）
    返回:
        bool: True 代表地面站可以看到该卫星, False 代表不可见
    """
    # 获取地面站和卫星的 ECEF 坐标
    G = lat_lon_alt_to_ecef(ground_lat_deg, ground_lon_deg, 0.0)  # (xg, yg, zg)
    S = lat_lon_alt_to_ecef(sat_lat_deg, sat_lon_deg, sat_alt_km)  # (xs, ys, zs)

    # 将它们转换为向量形式，便于点积运算
    Gx, Gy, Gz = G
    Sx, Sy, Sz = S

    # 向量 V = S - G
    Vx, Vy, Vz = (Sx - Gx, Sy - Gy, Sz - Gz)

    # 点积 V · G
    dot_VG = Vx * Gx + Vy * Gy + Vz * Gz

    # 如果点积大于0，说明向量V与向量G的夹角小于 90°，卫星在地平线上方
    return dot_VG > 0


if __name__ == '__main__':
    # 示例：地面站在北京(39.9°N, 116.4°E)，卫星假设在(40°N, 117°E)，海拔500km
    ground_lat = 28.1049
    ground_lon = 112.5710

    # 卫星示例位置
    sat_lat = -29.8398
    sat_lon = -50.5834
    sat_alt = 549.07  # km

    visible = is_satellite_visible_from_ground(
        ground_lat, ground_lon,
        sat_lat, sat_lon, sat_alt
    )

    if visible:
        print("卫星可覆盖到地面终端。")
    else:
        print("卫星不可覆盖该地面终端。")