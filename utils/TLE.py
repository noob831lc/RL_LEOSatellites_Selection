import os
import requests
from pathlib import Path
from datetime import datetime
from typing import Union
from skyfield.iokit import parse_tle_file
from skyfield.api import load
from skyfield.sgp4lib import EarthSatellite


# TLE文件的处理相关函数都在这里

# 获取Norad的TLE数据 可以自己换URL 主要网址是这个 https://celestrak.org/ 自己进去看

def request_tle_file(url: str = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle") -> \
        Union[requests.Response, Exception]:
    response = requests.get(url)
    
    if response.status_code == 200:
        print("成功获取 TLE 文件")
        return response
    else:
        raise Exception(f"无法获取 TLE 文件，状态码：{response.status_code}")


# 保存TLE数据到本地 目录名可以自己取
def save_tle_file(response: requests.Response, directory_name: str = "satellite_data") -> Path:
    tle_data = response.text
    if tle_data:
        # 获取当前时间戳并格式化
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # directory = "satellite_data"  # 替换为你指定的目录名
        directory = Path(directory_name)
        directory.mkdir(exist_ok=True)

        # 构造文件名
        file_path = f"{directory_name}/starlink_tle_{timestamp}.txt"

        # 将内容写入文件
        with open(file_path, "w") as file:
            file.write(tle_data)

        print(f"Data saved to {file_path}")
        return file_path


# 根据本地TLE数据获取卫星对象 返回的是卫星对象的列表
def parse_tle_data(tle_file_path: str) -> list[EarthSatellite]:
    ts = load.timescale()
    with load.open(tle_file_path) as f:
        satellites = list(parse_tle_file(f, ts))
    print(f'Loaded {len(satellites)} satellites')
    return satellites


# 注意 北京时间是UTC+8h 准确UTC时间是当前时间-8h
# 根据输入的UTC时间初步筛选掉不在时间范围内的卫星 （TLE+SGP4得到的轨道信息有时效性 每个TLE有个Epoch属性，就是卫星上传TLE到地面的一个时间,如果Epoch属性与当前时间差的很多,轨道数据误差会大）
def filter_epoch(
    satellites: list[EarthSatellite],
    utc_time: datetime,
    time_threshold: float = 1.0
) -> list[EarthSatellite]:

    ts = load.timescale()
    filter_time = ts.from_datetime(utc_time)

    selected_sats = list(filter(
        lambda sat: abs(sat.epoch - filter_time) <= time_threshold,
        satellites
    ))

    print(f"初步筛选后的卫星数量: {len(selected_sats)}")

    return selected_sats
