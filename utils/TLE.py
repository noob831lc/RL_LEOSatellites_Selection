import requests

from skyfield.iokit import parse_tle_file
from skyfield.api import load

from pathlib import Path
from datetime import datetime
from typing import Union
from skyfield.sgp4lib import EarthSatellite


timescale = load.timescale()
# 该模块用于获取卫星TLE数据

# 1. 从CelesTrak请求TLE文件
def request_tle_file(
    group: str = "starlink",
    format: str = "tle",
    base_url: str = "https://celestrak.org/NORAD/elements/gp.php"
) -> Union[requests.Response, Exception]:
    """
    Request satellite data from CelesTrak for a specific satellite group in the specified format.

    Args:
        group: Satellite group to request data for. Options include:
               'starlink' (default) - SpaceX Starlink satellites
               'oneweb' - OneWeb satellites
               'planet' - Planet Labs satellites
               'gps-ops' - GPS operational satellites
               'glo-ops' - GLONASS operational satellites
               'galileo' - Galileo satellites
               'beidou' - BeiDou satellites
               'active' - All active satellites
               'stations' - Space stations
               'visual' - 100 (or so) brightest satellites
               'weather' - Weather satellites
               'noaa' - NOAA satellites
               'goes' - GOES satellites
               'resource' - Earth resources satellites
               'sarsat' - Search & rescue satellites
               'dmc' - Disaster monitoring satellites
               'tdrss' - Tracking and data relay satellites
               'geo' - Geostationary satellites
               'intelsat' - Intelsat satellites
               'ses' - SES satellites
               'iridium' - Iridium satellites
               'iridium-NEXT' - Iridium NEXT satellites
               'orbcomm' - Orbcomm satellites
               'globalstar' - Globalstar satellites
               'amateur' - Amateur radio satellites
               'x-comm' - Experimental communications satellites
               'other-comm' - Other communications satellites
               'satnogs' - SatNOGS satellites
               'gorizont' - Gorizont satellites
               'raduga' - Raduga satellites
               'molniya' - Molniya satellites
               'military' - Military satellites (unclassified)
               'radar' - Radar calibration satellites
               'cubesat' - CubeSats
               'special' - Special interest satellites

        format: Data format to request. Options include:
               'tle' (default) - Two-Line Element Set
               '3le' - Three-Line Element Set
               '2le' - Two-Line Element Set with no title line
               'xml' or 'omm_xml' - OMM in XML format
               'kvn' or 'omm_kvn' - OMM in KVN format
               'json' - JSON format
               'json_pp' or 'json_pretty' - Pretty-printed JSON
               'csv' - CSV format

        base_url: Base URL for the CelesTrak API request (default: gp.php endpoint)

    Returns:
        Response object if successful, Exception if failed
    """
    # Format mapping to the actual format parameters used by CelesTrak
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

    # Get the correct format parameter or default to 'tle'
    format_param = format_map.get(format.lower(), 'tle')

    # Construct the URL with the group and format parameters
    full_url = f"{base_url}?GROUP={group}&FORMAT={format_param}"

    # Make the request
    response = requests.get(full_url)

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
    Save satellite data to a local file with a name based on the satellite group, format, and timestamp.

    Args:
        response: The response object from request_tle_file function
        group: The satellite group name that was requested
        format: The format of the satellite data
        directory_name: Directory to save the file in

    Returns:
        Path: The path to the saved file
    """
    data = response.text

    if not data:
        raise ValueError("Response contains no data to save")

    # Get current timestamp in a readable format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directory if it doesn't exist
    directory = Path(directory_name)
    directory.mkdir(exist_ok=True)

    # Format mapping to file extensions
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

    # Get the right file extension for the format
    extension = format_to_extension.get(format.lower(), 'txt')

    # Construct filename: group_format_timestamp.extension
    filename = f"{group}_{format}_{timestamp}.{extension}"
    file_path = directory / filename

    # Write the data to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(data)

    print(f"数据已保存至 {file_path}")
    return file_path


# 根据本地TLE数据获取卫星对象 返回的是卫星对象的列表
def parse_tle_data(tle_file_path: str) -> list[EarthSatellite]:
    
    if isinstance(tle_file_path, Path):
        tle_file_path = str(tle_file_path)
        
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
) -> list[EarthSatellite]:
    
    filter_time = timescale.from_datetime(utc_time)

    selected_sats = list(filter(
        lambda sat: abs(sat.epoch - filter_time) <= time_threshold_days,
        satellites
    ))

    print(f"初步筛选后的卫星数量: {len(selected_sats)}")

    return selected_sats
