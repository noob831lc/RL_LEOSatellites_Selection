import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt


################################################################################
# 1. 参数与辅助函数
################################################################################

def db2lin(db):
    """dB 转 线性比值"""
    return 10 ** (db / 10.0)


def lin2db(x):
    """线性比值 转 dB"""
    return 10 * np.log10(x + 1e-30)


def circ_shift(arr, shift):
    """
    在频域或者时域经常需要做循环移位，这里提供一个简易函数
    """
    return np.roll(arr, shift, axis=0)  # 对0维度作roll


def rms(x):
    """ root mean square """
    return np.sqrt(np.mean(np.abs(x) ** 2))


################################################################################
# 2. OTFS 调制与解调
################################################################################

def otfs_modulation_dd_to_tf(x_dd, M, N):
    """
    OTFS调制核心： 将DD域符号 x_dd(M×N) 作逆SFFT→得到TF域符号 X_tf(M×N)
    简化处理： 假设使用正交脉冲
    x_dd shape: (M, N), M为延迟维度，N为多普勒维度
    返回 X_tf shape: (M, N)
    """
    # ISFFT = iFFT w.r.t. Doppler (N维) + FFT w.r.t. Delay (M维)
    # 先对多普勒维度做 ifft，再对延迟维做 fft
    X_tf_temp = ifft(x_dd, axis=1)  # 对N维(多普勒)做IFFT
    X_tf = fft(X_tf_temp, axis=0)  # 对M维(延迟)做FFT
    return X_tf


def otfs_demodulation_tf_to_dd(Y_tf, M, N):
    """
    OTFS解调核心： 将TF域符号 Y_tf(M×N) 作SFFT→得到DD域符号 y_dd(M×N)
    Y_tf shape: (M, N)
    返回 y_dd shape: (M, N)
    """
    # SFFT = iFFT w.r.t. Delay + FFT w.r.t. Doppler
    y_temp = ifft(Y_tf, axis=0)  # 对M维(延迟)做IFFT
    y_dd = fft(y_temp, axis=1)  # 对N维(多普勒)做FFT
    return y_dd


def tf_to_time_domain(X_tf, M, N):
    """
    将TF域符号X_tf做二维IFFT(或先对一维做IFFT再对另一维做IFFT)，
    得到时域信号 x_time(M*N长度)。
    """
    # 对每个子载波分别做IFFT (行方向M) 然后再合并
    # 这里用最简单的二维IFFT代替更严格的OTFS脉冲成形
    x_time_2d = ifft(ifft(X_tf, axis=0), axis=1)
    # 拉直为一维信号
    x_time = x_time_2d.flatten()
    return x_time


def time_domain_to_tf(y_time, M, N):
    """
    将接收时域信号 y_time 转到TF域 Y_tf
    (与上面tf_to_time_domain相对应，做FFT2即可)
    """
    y_time_2d = np.reshape(y_time, (M, N))
    Y_tf = fft(fft(y_time_2d, axis=0), axis=1)
    return Y_tf


################################################################################
# 3. 导频嵌入与提取
################################################################################

def embed_pilot_in_dd(x_dd, pilot_amp, pilot_pos=(0, 0)):
    """
    在DD域的 x_dd(M×N) 中，埋一个导频pilot (假设仅1个pilot)，
    并把数据符号先置0以示简单。
    pilot_pos: (l, k) pilot位置
    pilot_amp: pilot幅度
    """
    x_dd_out = np.zeros_like(x_dd, dtype=complex)
    x_dd_out[pilot_pos[0], pilot_pos[1]] = pilot_amp
    return x_dd_out


def extract_pilot_from_dd(y_dd, pilot_pos=(0, 0)):
    """
    在DD域上提取pilot对应的接收位置的值
    """
    return y_dd[pilot_pos[0], pilot_pos[1]]


################################################################################
# 4. 多径分数多普勒信道模拟
################################################################################

def multipath_fractional_doppler_channel(x_time, fs, carrier_freq, paths):
    """
    多径分数多普勒信道模拟【简化版】：
    x_time: 发送时域信号
    fs:    采样率(与OTFS块长度/子载波间隔相关)
    carrier_freq: 载波频率(Hz)
    paths: 列表，每个元素是 (delay_in_samples, doppler_in_hz, path_gain)

    返回: y_time(叠加多径后的接收信号)

    说明：此处仅做“延迟+多普勒”乘性模拟，对每条路径做：
      y_i(t) = path_gain * x_time(t - delay) * e^{j2π doppler * t}
    并做简单的采样。
    """
    n_samples = len(x_time)
    y_time_total = np.zeros(n_samples, dtype=complex)
    t_idx = np.arange(n_samples) / fs  # 时间轴

    for (delay_samp, doppler_hz, gain) in paths:
        # 时域做一个延迟
        delayed_signal = np.zeros_like(x_time, dtype=complex)
        start_idx = delay_samp
        end_idx = n_samples
        if start_idx < end_idx:
            # 有效部分
            delayed_signal[start_idx:] = x_time[:(end_idx - start_idx)]

        # 多普勒调制
        doppler_phase = np.exp(1j * 2 * np.pi * doppler_hz * t_idx)
        delayed_signal_dopp = delayed_signal * doppler_phase

        # 路径增益
        path_out = gain * delayed_signal_dopp

        # 叠加到总接收信号上
        y_time_total += path_out

    return y_time_total


################################################################################
# 5. 阈值判决(Threshold-based)初步估计
################################################################################

def threshold_based_estimator_dd(y_dd, pilot_amp, threshold):
    """
    在DD域上对接收到的DD信号y_dd，进行门限判决，得到“有效DD域信道”估计:
      h_w_est[k,l] = y_dd[k,l] / pilot_amp, 如果 |y_dd[k,l]|>=threshold
                  = 0, 否则
    """
    M, N = y_dd.shape
    h_w_est = np.zeros((M, N), dtype=complex)
    for l in range(M):
        for k in range(N):
            if np.abs(y_dd[l, k]) >= threshold:
                h_w_est[l, k] = y_dd[l, k] / pilot_amp
            else:
                h_w_est[l, k] = 0.0
    return h_w_est


################################################################################
# 6. 单路径 & 多路径情况下的线性方程求解
################################################################################

def solve_single_path_doppler_gain(h_w_est, delay_idx, k_candidates):
    """
    单路径场景：从该延迟bin里选能量最大的两个采样点(k1, k2)，
    构造 2×2 线性方程组求解 (alpha, z)，进而得到分数多普勒 k_d_i 及路径增益。
    注意：只是简化演示，与论文中公式可能有 slight mismatch。
    """
    M, N = h_w_est.shape
    W = np.exp(-1j * 2 * np.pi / N)  # W = e^{-j2π/N}

    # 选出2个能量最大的采样
    mag_candidates = np.abs(h_w_est[delay_idx, k_candidates])
    if len(k_candidates) < 2:
        # 特殊情况，返回None
        return None, None

    top_idx = np.argsort(mag_candidates)[-2:]  # 取后2个最大的索引
    k1 = k_candidates[top_idx[0]]
    k2 = k_candidates[top_idx[1]]

    Gk1 = h_w_est[delay_idx, k1]
    Gk2 = h_w_est[delay_idx, k2]

    # 根据论文公式(简化形式):
    # [ Gk1 ] = [ 1, Gk1 * (W^k1) ] [ alpha ]
    # [ Gk2 ]   [ 1, Gk2 * (W^k2) ] [  z   ]
    # 这里使用最小二乘
    A = np.array([
        [1., Gk1 * (W ** k1)],
        [1., Gk2 * (W ** k2)]
    ], dtype=complex)
    b = np.array([Gk1, Gk2], dtype=complex)

    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    alpha_i, z_i = sol[0], sol[1]

    # 从z_i推分数多普勒
    # z_i = e^{ j 2π (kν+κν)/N } => k_d_i = arg(z_i)*N/(2π)
    kd_i = (np.angle(z_i) * N) / (2 * np.pi)

    # 路径增益(简单示例，实际应结合论文的(1 - z^N)等项)
    gain_i = alpha_i  # 仅作简化

    return kd_i, gain_i


def find_path_indices_in_delay_bin(h_w_est, delay_idx, energy_thresh):
    """
    在给定延迟 bin=delay_idx 中，找到所有能量超过阈值的位置，
    可能对应单条或多条路径。
    """
    row_slice = h_w_est[delay_idx, :]  # shape (N,)
    k_inds = np.where(np.abs(row_slice) >= energy_thresh)[0]
    return k_inds


################################################################################
# 7. 主流程(整合与示例)
################################################################################

def main_sim_otfs_fractional_doppler():
    np.random.seed(1234)

    # ========== 系统 & 模拟参数 ==========
    M, N = 32, 32  # OTFS维度(延迟×多普勒)
    subcarrier_spacing = 7.5e3  # 7.5 kHz
    T_symbol = 1.0 / subcarrier_spacing  # 符号长度 (approx)
    fs = M * subcarrier_spacing  # 采样率(非常简化的假设)
    carrier_freq = 3e9  # 载波3GHz
    pilot_amp_db = 20.0  # pilot功率(dB)，相对于数据=0dB
    pilot_amp = np.sqrt(db2lin(pilot_amp_db))  # 转线性振幅，使得它比数据符号大
    noise_power_db = -20.0  # 信道噪声功率(dB)
    noise_power_lin = db2lin(noise_power_db)

    # ========== 生成DD域发送符号（仅pilot）==========
    x_dd = np.zeros((M, N), dtype=complex)
    pilot_pos = (5, 8)  # 随意设一个pilot位置
    x_dd = embed_pilot_in_dd(x_dd, pilot_amp, pilot_pos=pilot_pos)

    # ========== 调制OTFS：DD->TF->时域 ==========
    X_tf = otfs_modulation_dd_to_tf(x_dd, M, N)
    x_time = tf_to_time_domain(X_tf, M, N)

    # ========== 构造多径+分数多普勒信道 ==========
    # 假设 P=3 径，每条径有 (delay_samp, doppler_hz, path_gain)
    # delay_samp  以“采样点”为单位
    # doppler_hz  真实多普勒频率 (例如 200Hz ~ 300Hz)
    # path_gain   复增益
    paths = [
        (0, 300.0, 1.0),  # 径1: 0采样延迟, 300Hz多普勒
        (2, -200.0, 0.8),  # 径2: 2采样延迟, -200Hz多普勒
        (8, 50.5, 0.6),  # 径3: 8采样延迟, 50.5Hz多普勒(带小数)
    ]
    # 通过该函数构造多径叠加
    y_time_chan = multipath_fractional_doppler_channel(x_time, fs, carrier_freq, paths)

    # ========== 加噪声 ==========
    noise = (np.random.randn(len(y_time_chan)) + 1j * np.random.randn(len(y_time_chan))) * np.sqrt(noise_power_lin / 2)
    y_time_rx = y_time_chan + noise

    # ========== 接收端OTFS解调：时域->TF->DD ==========
    Y_tf = time_domain_to_tf(y_time_rx, M, N)
    y_dd = otfs_demodulation_tf_to_dd(Y_tf, M, N)

    # ========== 阈值判决，初步估计DD域有效信道 ==========
    threshold_value = 0.01  # 门限(根据噪声水平等确定)
    h_w_est = threshold_based_estimator_dd(y_dd, pilot_amp, threshold_value)

    # ========== 尝试在所有延迟bin下做搜索，并演示单径分数多普勒估计 ==========
    # 注意：若某个延迟bin下可能有多条路径，就要考虑多路径的线性方程组
    # 这里仅做演示，实际需进一步区分多路径情况
    for l_idx in range(M):
        k_inds = find_path_indices_in_delay_bin(h_w_est, l_idx, energy_thresh=0.05)
        if len(k_inds) == 0:
            continue
        # 简化：默认假设该bin里只含一个主要路径
        kd_i, gain_i = solve_single_path_doppler_gain(h_w_est, l_idx, k_inds)
        if kd_i is not None:
            print(f"Delay bin l={l_idx}, found candidate path => k_d_i={kd_i:.3f}, gain={gain_i:.3f}")

    # ========== 结果可视化 (可选) ==========
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Magnitude of h_w_est in DD domain")
    plt.imshow(np.abs(h_w_est), aspect='auto', origin='lower')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Phase of h_w_est in DD domain")
    plt.imshow(np.angle(h_w_est), aspect='auto', origin='lower')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main_sim_otfs_fractional_doppler()