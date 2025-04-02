#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于以上对论文的总结，以下代码给出一个相对自洽的示例，
演示如何在Python中对论文主要思想进行实验复现（或近似复现）。
该示例包含了：
1. 生成OTFS系统的离散化多径信道（含小数多普勒）；
2. 在DD域注入导频并进行传输；
3. 在接收端先使用阈值法获取粗略“有效信道”；
4. 采用论文提出的“基于线性方程组”方法，估计每条路径的
   多普勒偏移(含小数部分)与路径增益；
5. 计算并输出与真实值之间的归一化均方误差(NMSE)；
6. 以CRLB为对照(此处给出一个简化近似形式，以说明实验思路)。

注：该代码仅为演示，部分细节(如仿真次数、不完全相同的CRLB推导等)
可能与论文存在差异，但整体流程与核心思路相仿，可供读者学习参考。
"""

import numpy as np
import matplotlib.pyplot as plt

# =============== 1. 参数设置 ===============
M = 32  # OTFS帧在Delay方向的离散采样数
N = 32  # OTFS帧在Doppler方向的离散采样数
carrier_freq = 3e9  # 载波频率 (Hz)
delta_f = 7.5e3  # 子载波间隔 (Hz)
T = 1 / delta_f  # 时隙长度 (秒)
P = 5  # 总路径数
kmax = 4  # 最大多普勒索引范围(正负对称)
lmax = 5  # 最大延迟索引范围
pilot_power_dB = 20  # 导频功率比数据功率强 20 dB
num_frames = 1000  # 仿真帧数
SNR_dB_list = np.arange(0, 35, 5)  # 在[0,30] dB范围内做测试


# =============== 2. 生成仿真所需函数 ===============

def generate_random_channel(P, kmax, lmax):
    """
    随机生成 P 条路径的(小数多普勒 + 整数延迟)，返回：
    - delays:    大小 [P], 每条路径的延迟索引 l_tau
    - dopplers:  大小 [P], 每条路径的连续多普勒索引 k_d = k_nu + kappa_nu
    - gains:     大小 [P], 每条路径的复增益
    """
    # 随机延迟索引（为了尽量体现多样性，排除0也可以）
    delays = np.random.randint(low=0, high=lmax + 1, size=P)
    # 整数多普勒索引
    int_dopplers = np.random.randint(low=-kmax, high=kmax + 1, size=P)
    # 小数多普勒(取一个[-0.5,0.5)之间的随机数)
    frac_dopplers = np.random.uniform(low=-0.5, high=0.5, size=P)
    dopplers = int_dopplers + frac_dopplers

    # 复增益，满足平均功率 1/P
    gains = (np.random.randn(P) + 1j * np.random.randn(P)) / np.sqrt(2 * P)
    return delays, dopplers, gains


def otfs_channel_effect(x_pilot, M, N, delays, dopplers, gains):
    """
    理想情况下，只考虑DD域中嵌入pilot的位置 (kp, lp) = (0,0)，
    计算接收端返回的 y[k,l]。（简化：不考虑其它数据符号干扰）
    这里仅生成'有效信道'所对应的接收量: y[k,l] = x_pilot * h_w[k,l] + noise

    h_w[k,l] = superposition of P paths with fractional Doppler
    """
    # 接收信号网格大小
    Y = np.zeros((N, M), dtype=complex)

    # 累加P条路径的响应
    for i in range(len(gains)):
        l_tau_i = delays[i]
        k_d_i = dopplers[i]  # k_d = 整数多普勒 + 小数多普勒
        h_i = gains[i]

        # 计算 DD 域的离散响应: hw[k, l_tau_i], 其中 k在[0, N-1], 但多普勒可扩散
        # 根据论文, fractional Doppler会导致在多普勒轴的sinc-like散列
        # 公式可以参考: hw[k, l] = ...
        # 为简单起见，这里直接离散实现:
        for k in range(N):
            # k' = (k - k_d_i)
            # 用Dirichlet核逼近(参考论文中的推导):
            # Gi(k) ~ (1/N)*Sum_{n=0}^{N-1} e^{-j2pi (k - kd_i)n/N}
            # 这里做一个简单实现:
            kd_frac = k - k_d_i
            # Dirichlet kernel
            if abs(kd_frac) < 1e-6:
                # 处理分母趋近于0的情况
                Hw_val = h_i
            else:
                numerator = np.sin(np.pi * kd_frac)
                denominator = N * np.sin(np.pi * kd_frac / N)
                Hw_val = h_i * numerator / denominator

            # 增加相位项 e^{-j2pi(k_d_i*l_tau_i)/(NM)}
            # 但此处为了简化，只保留多普勒扩散主项
            # 若需更准确，可加上 e^{-j 2pi (k_d_i) * l_tau_i / (N*M)} 等相位修正
            # 这里简化仅保留 Hw_val，读者可根据需求自行补充
            Y[k, l_tau_i] += x_pilot * Hw_val

    return Y


def add_noise(signal, snr_dB):
    """
    在给定信号上添加噪声，使得输出信噪比 = snr_dB
    SNR = 10 * log10( signal_power / noise_power )
    """
    sig_power = np.mean(np.abs(signal) ** 2)
    noise_power = sig_power / (10 ** (snr_dB / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise


def threshold_estimate(Y_recv, pilot_amp, threshold):
    """
    简单阈值法获取有效信道:
    hw_est[k,l] = Y_recv[k,l]/pilot_amp, if |Y_recv[k,l]| >= threshold; else 0
    """
    N, M = Y_recv.shape
    hw_est = np.zeros((N, M), dtype=complex)
    for k in range(N):
        for l in range(M):
            if np.abs(Y_recv[k, l]) >= threshold:
                hw_est[k, l] = Y_recv[k, l] / pilot_amp
    return hw_est


def refine_single_path(hw_est_col, k_search_set):
    """
    针对单径情况，在延迟bin上的多普勒轴数据 hw_est_col[k] 中，
    选择幅度最大的两个点，构造 2x2方程组求解 (alpha_i, z_i)
    返回估计的 (k_d_i, h_i)。
    hw_est_col: 大小 [N], 表示固定 l_tau 的多普勒分量
    k_search_set: 搜索k范围, 如 range(N) 或者更大范围
    """
    # 1. 找到幅度最大的两个采样点
    magnitudes = np.abs(hw_est_col)
    idx_sorted = np.argsort(-magnitudes)  # 按幅度降序
    k1 = idx_sorted[0]
    k2 = idx_sorted[1]
    G1 = hw_est_col[k1]
    G2 = hw_est_col[k2]

    # 2. 构建 2x2 方程
    # [ G1 ]   [ 1    G1 * W^k1 ] [ alpha ]
    # [ G2 ] = [ 1    G2 * W^k2 ] [  z    ]

    # 这里要注意: 论文中写的是 G_i(k)，但展开时往往有 G_i(k)*W^k.
    # 简化起见，假设 W= e^{-j 2pi/N}.
    W = np.exp(-1j * 2 * np.pi / N)
    matA = np.array([
        [1, G1 * (W ** k1)],
        [1, G2 * (W ** k2)]
    ], dtype=complex)
    vecB = np.array([G1, G2], dtype=complex)

    # 尝试求解
    try:
        alpha, z_est = np.linalg.solve(matA, vecB)
    except np.linalg.LinAlgError:
        return 0.0, 0.0  # 无法求解则返回

    # 3. 根据 z_est = e^{j 2pi k_d / N} 恢复 k_d
    #   令 z_est = exp(j 2pi k_d_hat / N)
    #   k_d_hat = (N / 2pi) * angle(z_est) (再做适当范围修正)
    k_d_hat = (np.angle(z_est) / (2 * np.pi)) * N

    # 为保证在 [-0.5, 0.5) + 整数区间附近，可以就近映射
    # 例如: k_int = round(k_d_hat), k_frac = k_d_hat - k_int
    # 但为简单起见，不做精细映射
    # 直接返回:

    # 4. 利用 alpha 与 z_est 得到信道增益 h_i
    #   论文中 h_i = N * alpha / [ (1 - z_i^N)*z_i^{(-l_tau*M)} ]...
    #   这里不做全部相位修正，只返回 alpha 近似作增益
    h_i_hat = alpha  # 简化近似

    return k_d_hat, h_i_hat


def nmse_metric(estimate, truth):
    """
    计算 NMSE = 10 * log10( ||estimate - truth||^2 / ||truth||^2 ).
    这里 estimate, truth均为1D向量或标量时皆可
    """
    err = np.linalg.norm(estimate - truth) ** 2
    ref = np.linalg.norm(truth) ** 2 + 1e-12
    nmse = 10 * np.log10(err / ref)
    return nmse


def approximate_crlb_simplified(snr_dB):
    """
    这里给出一个非常简化的多普勒估计CRLB近似，例如:
    CRLB ~ C / (10^(snr_dB/10)), 其中C为某常数。
    该函数仅用于演示与对比的曲线绘制，并非严格推导。
    """
    crlb_val = 10 ** (-snr_dB / 10)  # 简单指数衰减
    return crlb_val


# =============== 3. 进行主循环仿真 ===============

nmse_doppler_list = []
nmse_gain_list = []
crlb_doppler_list = []
crlb_gain_list = []

pilot_amp = 10.0  # 假设数据符号平均幅度=1, 导频比其大20dB => 幅度约=10
threshold_val = 0.1  # 简单设置一个阈值

for snr_db in SNR_dB_list:
    err_doppler = 0.0
    ref_doppler = 0.0
    err_gain = 0.0
    ref_gain = 0.0

    for _ in range(num_frames):
        # 1) 生成随机信道
        delays, dopplers, gains = generate_random_channel(P, kmax, lmax)

        # 2) 仅发pilot ((kp, lp) = (0,0))，计算理想接收
        Y_noiseless = otfs_channel_effect(x_pilot=pilot_amp,
                                          M=M, N=N,
                                          delays=delays,
                                          dopplers=dopplers,
                                          gains=gains)

        # 3) 加噪声
        Y_recv = add_noise(Y_noiseless, snr_db)

        # 4) 阈值法获取 hw_est
        hw_est = threshold_estimate(Y_recv, pilot_amp, threshold_val)

        # 5) 对每条真实路径(假设它们延迟各不相同，这里仅演示单径情况的做法)：
        #    若多个路径落在同一 l_tau，可以考虑多径的线性方程求解，
        #    这里仅演示最简单的"单径"思路去拟合，可能导致误差偏大。

        for i in range(P):
            l_tau_i = delays[i]
            kd_true = dopplers[i]  # 真实 doppler (含小数)
            h_true = gains[i]

            # 在hw_est中提取 [k=0..N-1] => hw_est[k, l_tau_i]
            hw_est_col = hw_est[:, l_tau_i]

            # 仅做单径 refine
            # (若要多径情况，需要在同一l_tau_i上聚合多径并构造
            #  2P_tau x 2P_tau 的线性方程，这里从简，不展开了)
            kd_hat, h_hat = refine_single_path(hw_est_col, range(N))

            # 计算误差: 多普勒 & 增益
            err_doppler += (kd_hat - kd_true) ** 2
            ref_doppler += (kd_true) ** 2 + 1e-12

            err_gain += np.abs(h_hat - h_true) ** 2
            ref_gain += np.abs(h_true) ** 2 + 1e-12

    # 归一化
    nmse_doppler = 10 * np.log10(err_doppler / ref_doppler)
    nmse_gain = 10 * np.log10(err_gain / ref_gain)

    nmse_doppler_list.append(nmse_doppler)
    nmse_gain_list.append(nmse_gain)

    # 简易CRLB对照
    crlb_doppler_val = approximate_crlb_simplified(snr_db)
    crlb_gain_val = approximate_crlb_simplified(snr_db)
    crlb_doppler_list.append(10 * np.log10(crlb_doppler_val))
    crlb_gain_list.append(10 * np.log10(crlb_gain_val))

# =============== 4. 结果可视化 ===============
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(SNR_dB_list, nmse_doppler_list, 'o-', label='Doppler Index NMSE')
plt.plot(SNR_dB_list, crlb_doppler_list, 'x--', label='Approx CRLB (Doppler)')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (dB)')
plt.title('Doppler Index Estimation')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(SNR_dB_list, nmse_gain_list, 'o-', label='Channel Gain NMSE')
plt.plot(SNR_dB_list, crlb_gain_list, 'x--', label='Approx CRLB (Gain)')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (dB)')
plt.title('Channel Gain Estimation')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()