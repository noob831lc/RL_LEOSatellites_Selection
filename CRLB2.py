import numpy as np
import matplotlib.pyplot as plt


def compute_fisher_matrix_singlepath(snr_linear, h_true, kappa_true):
    """
    示例性：根据(94)式等，对单径情形(P=1)的 Fisher 信息矩阵进行数值近似计算。
    这里仅用虚拟表达式演示，实际需结合 OTFS 参数(子载波数/导频位置/采样点等)。
    snr_linear: 线性域SNR，例如10^(SNR_dB / 10)
    h_true, kappa_true: 单径下真实参数 (可用于构造偏导数)
    返回 2x2 的 Fisher 信息矩阵
    """
    # 这是一个示例性的简单模拟，实际应结合(95)-(98)式做数值求导
    # 下方只给出一个随 snr_linear 成正比的“假”矩阵，做演示用
    I = np.array([[snr_linear * 0.8, snr_linear * 0.05],
                  [snr_linear * 0.05, snr_linear * 0.5]])
    return I


def compute_fisher_matrix_multipath(snr_linear, h_true_array, kappa_true_array):
    """
    示例性：根据(94)式等，对多径情形(P>1)的 Fisher 信息矩阵进行数值近似计算。
    snr_linear: 线性域SNR
    h_true_array, kappa_true_array: 分别为长度为P的增益、分数多普勒真值
    返回 2P x 2P 的 Fisher 信息矩阵
    """
    P = len(h_true_array)
    # 这里做一个“假设”构造：主对角线和部分非对角线元素与 snr_linear 成正比
    # 实际需根据(97)-(98)式对每条路径做偏导，累加后得到
    I = np.zeros((2 * P, 2 * P))
    for i in range(2 * P):
        # 主对角线示例：与 snr_linear, |h|^2 或其他因素正相关
        I[i, i] = (i + 1) * 0.5 * snr_linear
    # 简单加一些耦合项
    for i in range(2 * P - 1):
        I[i, i + 1] = 0.1 * snr_linear
        I[i + 1, i] = 0.1 * snr_linear
    return I


def compute_crlb(I_fisher):
    """
    输入: Fisher 信息矩阵 I_fisher
    输出: CRLB 矩阵(即 I_fisher 的逆)
    """
    I_inv = np.linalg.inv(I_fisher)
    return I_inv


def nmse_crlb_singlepath(I_inv, h_true, kappa_true):
    """
    根据(99)-(100)式等，计算单径下 h, kappa 的CRLB并转为NMSE(dB)。
    I_inv: 2x2 CRLB 矩阵
    h_true, kappa_true: 真值
    """
    # CRLB 对角线
    crlb_var_h = I_inv[0, 0]  # variance of h
    crlb_var_k = I_inv[1, 1]  # variance of kappa

    # 归一化(参考(99)-(100)式: 先把 h_true,kappa_true 视为向量做 ||.||^2 处理)
    norm_h2 = np.abs(h_true) ** 2
    norm_k2 = np.abs(kappa_true) ** 2

    nmse_h = 10 * np.log10(crlb_var_h / norm_h2)  # dB
    nmse_k = 10 * np.log10(crlb_var_k / norm_k2)  # dB

    return nmse_h, nmse_k


def nmse_crlb_multipath(I_inv, h_array, kappa_array):
    """
    多径下CRLB。I_inv为 2P x 2P，前P对应h，后P对应kappa。
    返回平均 NMSE(dB)，对应(99)-(100)式的hhat_bound 和 kappahat_bound(再转dB)
    """
    P = len(h_array)
    # 提取 CRLB 对角线
    crlb_h = np.diag(I_inv)[0:P]  # h相关
    crlb_k = np.diag(I_inv)[P:2 * P]  # kappa相关

    # 实际中 h,kappa 也可能是复增益等，需要作进一步处理
    norm_h2 = np.sum(np.abs(h_array) ** 2)
    norm_k2 = np.sum(np.abs(kappa_array) ** 2)

    # (99)-(100)式: 取对角元求和 / norm
    sum_crlb_h = np.sum(crlb_h)
    sum_crlb_k = np.sum(crlb_k)

    nmse_h = 10 * np.log10(sum_crlb_h / norm_h2)
    nmse_k = 10 * np.log10(sum_crlb_k / norm_k2)

    return nmse_h, nmse_k


# ------------------主程序：模拟绘图(仅绘制CRLB)------------------
def main_plot_crlb():
    # 设置仿真参数
    snr_db_list = np.arange(0, 31, 5)  # 0~30dB，步长5
    snr_linear_list = 10 ** (snr_db_list / 10)

    # 单径情形: 设定真值
    h_true_single = 1.0 + 0.0j  # 模拟的单径增益
    kappa_true_single = 0.2  # 分数多普勒
    nmse_h_single = []
    nmse_k_single = []

    for snr_lin in snr_linear_list:
        # 计算Fisher矩阵
        I_single = compute_fisher_matrix_singlepath(snr_lin, h_true_single, kappa_true_single)
        # 逆矩阵得到CRLB
        I_inv = compute_crlb(I_single)
        # 计算NMSE(dB)
        nh, nk = nmse_crlb_singlepath(I_inv, h_true_single, kappa_true_single)
        nmse_h_single.append(nh)
        nmse_k_single.append(nk)

    # 多径情形(示例 P=3)
    P = 3
    h_true_multi = np.array([1.0, 0.8 + 0.2j, 0.5 - 0.3j])  # 随意假设
    kappa_true_multi = np.array([0.2, -0.3, 0.1])
    nmse_h_multi = []
    nmse_k_multi = []

    for snr_lin in snr_linear_list:
        I_multi = compute_fisher_matrix_multipath(snr_lin, h_true_multi, kappa_true_multi)
        I_inv_multi = compute_crlb(I_multi)
        nh, nk = nmse_crlb_multipath(I_inv_multi, h_true_multi, kappa_true_multi)
        nmse_h_multi.append(nh)
        nmse_k_multi.append(nk)

    # 绘图: Fig.2(Fig.3)类似，只是可能单径/多径放两张图
    plt.figure(figsize=(7, 5))
    # 单径CRLB: h
    plt.plot(snr_db_list, nmse_h_single, 'o-', label='Single-Path CRLB (h)')
    # 单径CRLB: kappa
    plt.plot(snr_db_list, nmse_k_single, 'x-', label='Single-Path CRLB (kappa)')

    # 多径CRLB: h
    plt.plot(snr_db_list, nmse_h_multi, 's-', label='Multi-Path CRLB (h)')
    # 多径CRLB: kappa
    plt.plot(snr_db_list, nmse_k_multi, 'd-', label='Multi-Path CRLB (kappa)')

    plt.xlabel('SNR (dB)')
    plt.ylabel('NMSE (dB)')
    plt.title('CRLB for Single-path and Multi-path ')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main_plot_crlb()