import math


def deg2rad(degrees):
    return degrees * math.pi / 180


def calculate_slant_range(lat1, lon1, h1, lat2, lon2, h2):
    """
    计算地面站与卫星之间的俯距（地面站假设在海平面，h1=0）

    lat1, lon1: 地面站纬度和经度（度）
    h1: 地面站高度（km）
    lat2, lon2: 卫星纬度和经度（度）
    h2: 卫星高度（km）

    返回距离（km）
    """
    R_e = 6371  # 地球半径，单位km
    # 转换为弧度
    lat1_rad = deg2rad(lat1)
    lon1_rad = deg2rad(lon1)
    lat2_rad = deg2rad(lat2)
    lon2_rad = deg2rad(lon2)

    delta_lon = lon2_rad - lon1_rad

    # 计算中心角
    central_angle = math.acos(math.sin(lat1_rad) * math.sin(lat2_rad) +
                              math.cos(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))

    # 计算俯距
    d = math.sqrt(R_e ** 2 + (R_e + h2) ** 2 - 2 * R_e * (R_e + h2) * math.cos(central_angle))
    return d


def calculate_fspl(d, f):
    """
    计算自由空间路径损耗（FSPL）

    d: 距离（km）
    f: 频率（GHz）

    返回FSPL（dB）
    """
    fspl = 20 * math.log10(d) + 20 * math.log10(f) + 92.45
    return fspl


def calculate_received_power(P_t_dBm, G_t, G_r, fspl, L_other=2):
    """
    计算接收信号功率（dBm）

    P_t_dBm: 发射功率（dBm）
    G_t: 发射天线增益（dBi）
    G_r: 接收天线增益（dBi）
    fspl: 自由空间路径损耗（dB）
    L_other: 其他损耗（dB），默认2 dB

    返回P_r（dBm）
    """
    P_r = P_t_dBm + G_t + G_r - fspl - L_other
    return P_r


def calculate_noise_power(T_s=290, B=10e6):
    """
    计算系统噪声功率（dBm）

    T_s: 系统噪声温度（K）
    B: 带宽（Hz）

    返回N（dBm）
    """
    k_dBm_Hz = -174  # 玻尔兹曼常数，单位dBm/Hz
    N = k_dBm_Hz + 10 * math.log10(B) + 10 * math.log10(T_s)
    return N


def calculate_c_n(P_r, N):
    """
    计算载噪比（C/N）（dB）

    P_r: 接收信号功率（dBm）
    N: 噪声功率（dBm）

    返回C/N（dB）
    """
    C_N = P_r - N
    return C_N


def convert_watt_to_dBm(P_watt):
    """
    将瓦特转换为dBm
    """
    if P_watt == 0:
        return -math.inf
    return 10 * math.log10(P_watt * 1000)


def main():
    # 示例参数（可根据实际情况修改）
    # 卫星位置
    sat_lat = 40.0  # 卫星纬度（度）
    sat_lon = -74.0  # 卫星经度（度）
    sat_alt = 550  # 卫星高度（km）

    # 地面站位置
    gs_lat = 34.0  # 地面站纬度（度）
    gs_lon = -118.0  # 地面站经度（度）
    gs_alt = 0  # 地面站高度（km），假设在海平面

    # 链路参数
    P_t_watt = 25  # 发射功率（瓦特）
    G_t = 35  # 发射天线增益（dBi）
    G_r = 20  # 接收天线增益（dBi）
    frequency = 2  # 频率（GHz）
    L_other = 2  # 其他损耗（dB）

    # 噪声参数
    T_s = 290  # 系统噪声温度（K）
    B = 10e6  # 带宽（Hz）

    # 链路需求
    required_C_N = 5  # 最低载噪比要求（dB）

    # 计算俯距
    d = calculate_slant_range(gs_lat, gs_lon, gs_alt, sat_lat, sat_lon, sat_alt)
    print(f"俯距 (d): {d:.2f} km")

    # 计算自由空间路径损耗
    fspl = calculate_fspl(d, frequency)
    print(f"自由空间路径损耗 (FSPL): {fspl:.2f} dB")

    # 将发射功率转换为dBm
    P_t_dBm = convert_watt_to_dBm(P_t_watt)
    print(f"发射功率 (P_t): {P_t_dBm:.2f} dBm")

    # 计算接收功率
    P_r = calculate_received_power(P_t_dBm, G_t, G_r, fspl, L_other)
    print(f"接收信号功率 (P_r): {P_r:.2f} dBm")

    # 计算噪声功率
    N = calculate_noise_power(T_s, B)
    print(f"噪声功率 (N): {N:.2f} dBm")

    # 计算载噪比
    C_N = calculate_c_n(P_r, N)
    print(f"载噪比 (C/N): {C_N:.2f} dB")

    # 判断是否接收
    if C_N >= required_C_N:
        print("信号被成功接收。")
    else:
        print("信号未被接收。")


if __name__ == "__main__":
    main()