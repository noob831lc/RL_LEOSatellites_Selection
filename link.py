import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LinkParameters:
    """链路参数数据类"""
    frequency: float  # GHz
    orbit_height: float  # km
    elevation_angle: float  # degrees
    tx_power: float  # W
    tx_antenna_gain: float  # dBi
    rx_antenna_gain: float  # dBi
    system_temp: float  # K
    data_rate: float  # bps
    modulation: str
    coding: str
    availability: float  # %


class LinkBudgetCalculator:
    """链路预算计算器类"""

    def __init__(self, params: LinkParameters):
        self.params = params
        self.results = {}

    def calculate_distance(self) -> float:
        """计算卫星到地面站的距离"""
        Re = 6371  # 地球半径(km)
        h = self.params.orbit_height
        el = math.radians(self.params.elevation_angle)

        distance = math.sqrt((Re + h) ** 2 - (Re * math.cos(el)) ** 2) - Re * math.sin(el)
        return distance

    def calculate_free_space_loss(self) -> float:
        """计算自由空间损耗"""
        distance = self.calculate_distance()
        freq_mhz = self.params.frequency * 1000
        fsl = 32.45 + 20 * math.log10(freq_mhz) + 20 * math.log10(distance)
        return fsl

    def calculate_atmospheric_loss(self) -> float:
        """计算大气损耗 (简化模型)"""
        if self.params.frequency < 10:
            return 0.1
        elif self.params.frequency < 20:
            return 0.2
        else:
            return 0.3

    def calculate_rain_loss(self) -> float:
        """计算雨衰损耗 (简化模型)"""
        # 基于频率和可用性的简化计算
        if self.params.frequency < 10:
            base_loss = 1.0
        else:
            base_loss = 2.0

        availability_factor = (100 - self.params.availability) / 10
        return base_loss * availability_factor

    def calculate_eirp(self) -> float:
        """计算EIRP"""
        tx_power_dbw = 10 * math.log10(self.params.tx_power)
        return tx_power_dbw + self.params.tx_antenna_gain

    def calculate_gt(self) -> float:
        """计算G/T"""
        return self.params.rx_antenna_gain - 10 * math.log10(self.params.system_temp)

    def calculate_cn0(self) -> float:
        """计算C/N0"""
        eirp = self.calculate_eirp()
        path_loss = (
                self.calculate_free_space_loss() +
                self.calculate_atmospheric_loss() +
                self.calculate_rain_loss()
        )
        gt = self.calculate_gt()

        cn0 = eirp - path_loss + gt + 228.6  # 228.6 is -10*log(k), where k is Boltzmann's constant
        return cn0

    def calculate_eb_n0(self) -> float:
        """计算Eb/N0"""
        cn0 = self.calculate_cn0()
        data_rate_db = 10 * math.log10(self.params.data_rate)
        return cn0 - data_rate_db

    def calculate_link_margin(self) -> float:
        """计算链路裕度"""
        eb_n0 = self.calculate_eb_n0()
        required_eb_n0 = self.get_required_eb_n0()
        return eb_n0 - required_eb_n0

    def get_required_eb_n0(self) -> float:
        """获取所需的Eb/N0 (基于调制和编码方式)"""
        modulation_coding_requirements = {
            "QPSK": {
                "1/2": 4.0,
                "3/4": 6.0
            },
            "8PSK": {
                "2/3": 8.0,
                "3/4": 9.0
            }
        }
        return modulation_coding_requirements.get(self.params.modulation, {}).get(self.params.coding, 6.0)

    def calculate_full_budget(self) -> Dict:
        """计算完整的链路预算"""
        try:
            self.results = {
                "distance_km": round(self.calculate_distance(), 2),
                "eirp_dbw": round(self.calculate_eirp(), 2),
                "free_space_loss_db": round(self.calculate_free_space_loss(), 2),
                "atmospheric_loss_db": round(self.calculate_atmospheric_loss(), 2),
                "rain_loss_db": round(self.calculate_rain_loss(), 2),
                "gt_db_k": round(self.calculate_gt(), 2),
                "cn0_db_hz": round(self.calculate_cn0(), 2),
                "eb_n0_db": round(self.calculate_eb_n0(), 2),
                "link_margin_db": round(self.calculate_link_margin(), 2)
            }
            return self.results
        except Exception as e:
            logger.error(f"计算错误: {str(e)}")
            raise

    def print_results(self):
        """打印格式化的结果"""
        if not self.results:
            self.calculate_full_budget()

        print("\n=== 链路预算计算结果 ===")
        print(f"卫星距离: {self.results['distance_km']:.2f} km")
        print("\n--- 发射端 ---")
        print(f"EIRP: {self.results['eirp_dbw']:.2f} dBW")
        print("\n--- 传播损耗 ---")
        print(f"自由空间损耗: {self.results['free_space_loss_db']:.2f} dB")
        print(f"大气损耗: {self.results['atmospheric_loss_db']:.2f} dB")
        print(f"雨衰损耗: {self.results['rain_loss_db']:.2f} dB")
        print("\n--- 接收端 ---")
        print(f"G/T: {self.results['gt_db_k']:.2f} dB/K")
        print("\n--- 链路性能 ---")
        print(f"C/N0: {self.results['cn0_db_hz']:.2f} dB-Hz")
        print(f"Eb/N0: {self.results['eb_n0_db']:.2f} dB")
        print(f"链路裕度: {self.results['link_margin_db']:.2f} dB")


def run_example():
    # 典型LEO卫星场景参数
    params = LinkParameters(
        frequency=12.0,  # GHz
        orbit_height=550.0,  # km
        elevation_angle=30.0,  # degrees
        tx_power=20.0,  # W
        tx_antenna_gain=30.0,  # dBi
        rx_antenna_gain=35.0,  # dBi
        system_temp=290.0,  # K
        data_rate=100e6,  # 100 Mbps
        modulation="QPSK",
        coding="1/2",
        availability=99.9  # %
    )

    calculator = LinkBudgetCalculator(params)
    calculator.print_results()

    # 不同仰角的计算
    angles = [10, 30, 60, 90]
    print("\n=== 不同仰角比较 ===")
    for angle in angles:
        params.elevation_angle = angle
        calc = LinkBudgetCalculator(params)
        results = calc.calculate_full_budget()
        print(f"\n仰角 {angle}°:")
        print(f"距离: {results['distance_km']:.2f} km")
        print(f"总损耗: {results['free_space_loss_db'] + results['atmospheric_loss_db'] + results['rain_loss_db']:.2f} dB")
        print(f"链路裕度: {results['link_margin_db']:.2f} dB")


if __name__ == "__main__":
    run_example()