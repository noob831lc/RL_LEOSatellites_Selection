#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def crlb_multi_path_doppler(N, T, sigma2, alphas, nus):
    """
    计算多径情形CRLB(每条路径都估计复增益和多普勒).

    参数说明:
    ----------
    N       : 观测样本数
    T       : 采样间隔
    sigma2  : 噪声方差
    alphas  : list/array, shape=(P,), 每条路径的复增益
    nus     : list/array, shape=(P,), 每条路径的多普勒

    返回:
    ----------
    CRLB_matrix : (3P,3P) 维numpy数组, 费舍尔信息矩阵的逆
    """
    P = len(alphas)
    # total param dimension = 3P
    # 我们先构造对每条路径的 partial_mu/partial_params, 然后把它们拼接起来

    n_idx = np.arange(N)

    # 对 real-valued param => Jacobian 扩展到(2N, 3P)
    J_real = np.zeros((2 * N, 3 * P), dtype=float)

    for p_idx in range(P):
        alpha_p = alphas[p_idx]
        nu_p = nus[p_idx]
        exp_part = np.exp(1j * 2 * np.pi * nu_p * n_idx * T)  # shape (N,)

        # 对第p_idx条路径，第p_idx组参数对应表：
        # => col offset in J_real = 3*p_idx
        col0 = 3 * p_idx

        # partial wrt Re(alpha_p) => exp_part
        # partial wrt Im(alpha_p) => j*exp_part
        # partial wrt nu_p       => alpha_p * j*2π*n*T * exp_part
        dmu_dRe = exp_part
        dmu_dIm = 1j * exp_part
        dmu_dNu = alpha_p * (1j * 2 * np.pi * n_idx * T) * exp_part

        # stack real and imag
        # For the partial wrt Re(alpha_p):
        # row0 = real part, row1 = imag part
        # shape for single param => (N,) => will place in J_real(0:N, col), J_real(N:2N, col)
        J_real[0:N, col0 + 0] = np.real(dmu_dRe)  # real part
        J_real[N:2 * N, col0 + 0] = np.imag(dmu_dRe)  # imag part

        J_real[0:N, col0 + 1] = np.real(dmu_dIm)
        J_real[N:2 * N, col0 + 1] = np.imag(dmu_dIm)

        J_real[0:N, col0 + 2] = np.real(dmu_dNu)
        J_real[N:2 * N, col0 + 2] = np.imag(dmu_dNu)

    # 费舍尔信息矩阵
    FIM = (1.0 / sigma2) * (J_real.T @ J_real)  # shape (3P,3P)
    CRLB_matrix = np.linalg.inv(FIM)  # (3P,3P)
    return CRLB_matrix


if __name__ == "__main__":
    # 测试多径CRLB
    N_test = 64
    T_test = 1e-3
    sigma2_test = 1.0

    # 假设有2条路径
    alphas_test = np.array([1.0 + 1.0j, 0.5 - 0.8j], dtype=complex)
    nus_test = np.array([5.0, 10.0])  # 5Hz / 10Hz

    crlb_mat = crlb_multi_path_doppler(N_test, T_test, sigma2_test, alphas_test, nus_test)
    print("Multi-path CRLB matrix shape:", crlb_mat.shape)
    print("CRLB matrix = \n", crlb_mat)
    # 每个路径3维, 共有2条路径 => 3*2=6维
    # crlb_mat[i,i] (对角元素) => 估计相应参数的方差下界
    # i=0 => Re(alpha1), i=1 => Im(alpha1), i=2 => nu1, i=3 => Re(alpha2), i=4 => Im(alpha2), i=5 => nu2
    print("\nDiagonal of CRLB =>", np.diag(crlb_mat))