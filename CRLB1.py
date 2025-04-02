#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def crlb_single_path_doppler(N, T, sigma2, alpha, nu):
    """
    计算单径情形下(复增益 + 多普勒)参数的CRLB。

    参数说明:
    ----------
    N      : 观测样本数
    T      : 每个采样点的时间间隔
    sigma2 : 噪声方差
    alpha  : 路径复增益 (complex)
    nu     : 多普勒频移(浮点, 单位Hz或归一化频偏)

    返回:
    ----------
    crlb : shape=(3,) 的numpy数组
       [Var(Re(alpha)), Var(Im(alpha)), Var(nu)]
    """
    # 构造观测序列下，对参数 (Re(α), Im(α), ν) 的雅可比矩阵
    n_idx = np.arange(N)
    exp_part = np.exp(1j * 2.0 * np.pi * nu * n_idx * T)

    # J(n, paramIndex)，paramIndex＝0:Re(alpha),1:Im(alpha),2:nu
    # 先用复数形式，再展开成实数域
    J = np.zeros((N, 3), dtype=np.complex128)
    # 对 Re(alpha)
    J[:, 0] = exp_part
    # 对 Im(alpha)
    J[:, 1] = 1j * exp_part
    # 对 nu
    # partial wrt nu => alpha * j*2π*n*T * exp(j*2π*nu*n*T)
    J[:, 2] = alpha * (1j * 2.0 * np.pi * n_idx * T) * exp_part

    # 实数展开(2N x 3)
    # row前N行为实部, 后N行为虚部
    J_real = np.vstack((np.real(J), np.imag(J)))  # shape=(2N,3)

    # 费舍尔信息矩阵(FIM) = (1 / sigma^2) * (J_real^T @ J_real)
    FIM = (1.0 / sigma2) * (J_real.T @ J_real)  # shape=(3,3)
    CRLB_matrix = np.linalg.inv(FIM)  # shape=(3,3)

    # 返回对角线: [Var(Re(alpha)), Var(Im(alpha)), Var(nu)]
    return np.diag(CRLB_matrix)


def main():
    # 参数设置
    N = 64  # 观测点数
    T = 1e-3  # 采样间隔(1ms)
    alpha = 1.0 + 1.0j  # 路径复增益
    nu = 10.0  # 多普勒(Hz)，可根据需要修改

    # SNR范围(dB)
    snr_db_list = np.linspace(0, 30, 30)  # 0~30 dB，共16个点

    # 用于保存计算结果
    var_re_alpha = []
    var_im_alpha = []
    var_nu = []

    for snr_db in snr_db_list:
        # 根据 SNR 计算噪声方差:
        # SNR = 10^(SNR_dB/10) = (信号功率) / (噪声功率)
        # 若假定信号功率 ~ |alpha|^2，则 sigma^2 = |alpha|^2 / SNR_(线性)
        snr_linear = 10.0 ** (snr_db / 10.0)
        signal_power = np.abs(alpha) ** 2  # 这里仅作为演示，实际可根据系统定义调整
        sigma2 = signal_power / snr_linear

        crlb = crlb_single_path_doppler(N, T, sigma2, alpha, nu)

        var_re_alpha.append(crlb[0])
        var_im_alpha.append(crlb[1])
        var_nu.append(crlb[2])

    # 转numpy便于绘图
    var_re_alpha = np.array(var_re_alpha)
    var_im_alpha = np.array(var_im_alpha)
    var_nu = np.array(var_nu)
    
    # 绘图
    plt.figure(figsize=(7, 5))

    # 使用对数刻度(可选)
    plt.semilogy(snr_db_list, var_nu, '^-', label='CRLB Var(nu)')

    plt.xlabel("SNR (dB)")
    plt.ylabel("Variance (log scale)")
    plt.title("CRLB vs. SNR for Single-Path Doppler Estimation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()