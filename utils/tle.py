import requests

from skyfield.iokit import parse_tle_file
from skyfield.api import load
from skyfield.sgp4lib import EarthSatellite
from pathlib import Path
from datetime import datetime
from typing import Union

timescale = load.timescale()


# 1. 从CelesTrak请求TLE文件
def request_tle_file(
    group: str = "starlink",
    format: str = "tle",
    base_url: str = "https://celestrak.org/NORAD/elements/gp.php"
) -> Union[requests.Response, Exception]:
    """
    从CelesTrak请求特定卫星组的卫星数据，以指定的格式返回。

    参数:
        group: 请求数据的卫星组。可选项包括:
               'starlink' (默认) - SpaceX Starlink卫星
               'oneweb' - OneWeb卫星
               'planet' - Planet Labs卫星
               'gps-ops' - GPS运行卫星
               'glo-ops' - GLONASS运行卫星
               'galileo' - Galileo卫星
               'beidou' - 北斗卫星
               'active' - 所有活动卫星
               'stations' - 空间站
               'visual' - 100个(左右)最亮的卫星
               'weather' - 气象卫星
               'noaa' - NOAA卫星
               'goes' - GOES卫星
               'resource' - 地球资源卫星
               'sarsat' - 搜救卫星
               'dmc' - 灾害监测卫星
               'tdrss' - 跟踪和数据中继卫星
               'geo' - 地球同步卫星
               'intelsat' - Intelsat卫星
               'ses' - SES卫星
               'iridium' - Iridium卫星
               'iridium-NEXT' - Iridium NEXT卫星
               'orbcomm' - Orbcomm卫星
               'globalstar' - Globalstar卫星
               'amateur' - 业余无线电卫星
               'x-comm' - 实验性通信卫星
               'other-comm' - 其他通信卫星
               'satnogs' - SatNOGS卫星
               'gorizont' - Gorizont卫星
               'raduga' - Raduga卫星
               'molniya' - Molniya卫星
               'military' - 军事卫星（未分类）
               'radar' - 雷达校准卫星
               'cubesat' - CubeSats
               'special' - 特殊兴趣卫星

        format: 请求的数据格式。可选项包括:
               'tle' (默认) - 两行元素集
               '3le' - 三行元素集
               '2le' - 无标题行的两行元素集
               'xml' 或 'omm_xml' - XML格式的OMM
               'kvn' 或 'omm_kvn' - KVN格式的OMM
               'json' - JSON格式
               'json_pp' 或 'json_pretty' - 格式化的JSON
               'csv' - CSV格式

        base_url: CelesTrak API请求的基本URL（默认: gp.php端点）

    返回:
        如果成功，返回响应对象；如果失败，返回异常
    """
    # 格式映射到CelesTrak实际使用的格式参数
    format_map = {
        'tle': 'tle',
        '3le': '3le',
        '2le': '2le',
        'xml': 'xml',
        'omm_xml': 'xml',
        'kvn': 'kvn',
        'omm_kvn': 'kvn',
        'json': 'json',
        'json_pp': 'json_pp',
        'json_pretty': 'json_pp',
        'csv': 'csv'
    }

    # 获取正确的格式参数，如果没有指定则默认为'tle'
    format_param = format_map.get(format.lower(), 'tle')

    # 构建带有组和格式参数的URL
    full_url = f"{base_url}?GROUP={group}&FORMAT={format_param}"

    # 发起请求
    response = requests.get(full_url,timeout=60)

    if response.status_code == 200:
        print(f"成功获取 {group}卫星数据\t格式：{format_param}")
        return response
    else:
        raise Exception(f"无法获取卫星数据，状态码：{response.status_code}, URL: {full_url}")


def save_tle_file(
    response: requests.Response,
    group: str = "starlink",
    format: str = "tle",
    directory_name: str = "satellite_data"
) -> Path:
    """
    将卫星数据保存到本地文件
    参数:
        response: request_tle_file 函数的响应对象
        group: 请求卫星组的名称
        format: 卫星数据的格式
        directory_name: 保存文件的目录

    返回:
        Path: 保存文件的路径
    """
    data = response.text

    if not data:
        raise ValueError("响应中没有要保存的数据")

    # 获取当前时间戳的可读格式
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 如果目录不存在，则创建目录
    directory = Path(directory_name)
    directory.mkdir(exist_ok=True)

    # 格式映射到文件扩展名
    format_to_extension = {
        'tle': 'txt',
        '3le': 'txt',
        '2le': 'txt',
        'xml': 'xml',
        'omm_xml': 'xml',
        'kvn': 'txt',
        'omm_kvn': 'txt',
        'json': 'json',
        'json_pp': 'json',
        'json_pretty': 'json',
        'csv': 'csv'
    }

    # 获取格式对应的文件扩展名
    extension = format_to_extension.get(format.lower(), 'txt')

    # 构造文件名: group_format_timestamp.extension
    filename = f"{group}_{format}_{timestamp}.{extension}"
    file_path = directory / filename

    # 将数据写入文件
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(data)

    print(f"数据已保存至 {file_path}")
    return file_path




def parse_tle_data(tle_file_path: str) -> list[EarthSatellite]:
    if isinstance(tle_file_path, Path):
        tle_file_path = str(tle_file_path) # Convert Path to string
    with load.open(tle_file_path) as f:
        satellites = list(parse_tle_file(f, timescale))
    print(f'Loaded {len(satellites)} satellites')
    return satellites



# 注意 北京时间是UTC+8h 准确UTC时间是当前时间-8h
# 根据输入的UTC时间初步筛选掉不在时间范围内的卫星 （TLE+SGP4得到的轨道信息有时效性 每个TLE有个Epoch属性，就是卫星上传TLE到地面的一个时间,如果Epoch属性与当前时间差的很多,轨道数据误差会大）
def filter_epoch(
    satellites: list[EarthSatellite],
    utc_time: datetime,
    time_threshold_days: float = 1.0
) -> list[EarthSatellite]: # filter_epoch
    filter_time = timescale.from_datetime(utc_time)

    selected_sats = list(filter(
        lambda sat: abs(sat.epoch - filter_time) <= time_threshold_days,
        satellites
    ))

    print(f"初步筛选后的卫星数量: {len(selected_sats)}")

    return selected_sats
