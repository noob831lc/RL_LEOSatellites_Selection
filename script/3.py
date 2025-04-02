import io, base64
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

###############################################################################
# 以下演示性代码并非论文原文的全部细节实现，但涵盖核心过程：
# 1) BPSK 在 AWGN 下的时延 / 多普勒估计性能
# 2) OTFS 在 AWGN 下的时延 / 多普勒估计性能
# 3) 简化的双选择性信道(多径 + 多普勒扩展)下，二者的延迟-多普勒响应可视化
# 4) 简要绘制多径误差包络(MEE)示例
#
# 读者可在此基础上修改带宽 B、子载波数 M、OFDM 符号数 N、采样率 Fs 等参数，
# 并构造相应的传播信道模型(CIR)进行更符合论文的性能复现。
###############################################################################


###############################################################################
# 1. 参数配置
###############################################################################
SNR_dBs = np.arange(0, 41, 5)  # 0~40 dB
N_experiments = 2000  # 每个 SNR 下的蒙特卡洛试验次数(可视需要加大)

# 信号参数(演示简化)
Fs = 1e6  # 采样率(Hz), 演示用
B = 1e6  # 信号带宽 ~ Fs
Tc = 1e-3  # 信号时长(秒), 用于BPSK/OTFS的一帧长度等
fc = 1e9  # 载频1GHz (LEO场景更可能是几 GHz, 仅示例数字)
true_delay = 5e-6  # 真实时延 5微秒 (对应~1500米距离)
true_dopp = 200.0  # 真实多普勒 200Hz (LEO可远大于此)


###############################################################################
# 2. 生成BPSK信号 并在AWGN中估计时延/多普勒
###############################################################################
def generate_bpsk_signal(num_samples=1000):
    # 生成简单的BPSK基带序列
    bits = np.random.randint(0, 2, num_samples)
    symbols = 2 * bits - 1  # 映射到{-1, +1}
    return symbols.astype(np.complex64)


def apply_channel_bpsk(tx_sig, delay_s, doppler_hz, snr_db):
    # 先在时域上做一个简单的延迟 & 多普勒处理
    # 为简便，这里只做离散采样延迟处理，若 delay_s 不是整数倍采样间隔则做插值
    # 多普勒只考虑调制相位增量。
    dt = 1.0 / Fs
    n = np.arange(len(tx_sig))

    # 多普勒相位
    phase = 2 * np.pi * doppler_hz * n * dt

    # 进行子样级延迟
    delay_samples = int(np.round(delay_s * Fs))
    rx_len = len(tx_sig) + delay_samples
    rx_sig = np.zeros(rx_len, dtype=np.complex64)
    if delay_samples < rx_len:
        rx_sig[delay_samples:] = tx_sig * np.exp(1j * phase[:len(tx_sig)])
    # 加噪声
    signal_power = np.mean(np.abs(rx_sig) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(rx_len) + 1j * np.random.randn(rx_len))
    rx_sig += noise
    return rx_sig


def estimate_delay_doppler_bpsk(rx_sig, tx_sig):
    # 这里用二维相关(相关函数)来做一个简化的 brute-force。
    # 实际中会做分段FFT加速，这里演示用。
    # 在论文中则需要更精细的搜索及采样插值。

    max_delay_search = 50  # 仅搜索前50个采样点
    doppler_search = np.linspace(-400, 400, 81)  # -400~400 Hz
    best_metric = -1e12
    est_delay = 0
    est_dopp = 0
    for dly in range(max_delay_search):
        if dly + len(tx_sig) > len(rx_sig):
            break
        segment = rx_sig[dly:dly + len(tx_sig)]
        for df in doppler_search:
            phase = np.exp(-1j * 2 * np.pi * df * np.arange(len(tx_sig)) / Fs)
            metric = np.abs(np.sum(segment * np.conj(tx_sig * phase)))
            if metric > best_metric:
                best_metric = metric
                est_delay = dly
                est_dopp = df
    # 转回实际物理量
    delay_est_s = est_delay / Fs
    doppler_est_hz = est_dopp
    return delay_est_s, doppler_est_hz


# 进行仿真BPSK
delay_rmse_bpsk = []
doppler_rmse_bpsk = []

for snr in SNR_dBs:
    d_errors = []
    f_errors = []
    for _ in range(N_experiments):
        tx_bpsk = generate_bpsk_signal(num_samples=2000)
        rx_bpsk = apply_channel_bpsk(tx_bpsk, true_delay, true_dopp, snr)
        d_hat, f_hat = estimate_delay_doppler_bpsk(rx_bpsk, tx_bpsk)
        d_errors.append(d_hat - true_delay)
        f_errors.append(f_hat - true_dopp)
    delay_rmse_bpsk.append(np.sqrt(np.mean(np.array(d_errors) ** 2)) * 3e8)  # 换算到米
    doppler_rmse_bpsk.append(np.sqrt(np.mean(np.array(f_errors) ** 2)))


###############################################################################
# 3. 简化OTFS在AWGN下的仿真 (示意)
###############################################################################
# 这里只做极为简化的“OTFS”生成并相关搜索(更多细节取决于N, M, ISFFT等)
# 在真正大规模仿真中，需要完整的OTFS调制/解调流程 + oversample DD 域。
###############################################################################
def generate_otfs_pilot(num_symbols=2000):
    # 简化：相当于发一段全1序列或DD域里仅一个非零点
    # 真正OTFS应构建DD网格, IFFT/FFT; 这里只演示波形
    return np.ones(num_symbols, dtype=np.complex64)


def apply_channel_otfs(tx_sig, delay_s, doppler_hz, snr_db):
    # 类似的简易延迟、多普勒处理 + AWGN
    dt = 1.0 / Fs
    n = np.arange(len(tx_sig))
    phase = 2 * np.pi * doppler_hz * n * dt
    delay_samples = int(round(delay_s * Fs))
    rx_len = len(tx_sig) + delay_samples
    rx_sig = np.zeros(rx_len, dtype=np.complex64)
    if delay_samples < rx_len:
        rx_sig[delay_samples:] = tx_sig * np.exp(1j * phase[:len(tx_sig)])
    # 加噪
    signal_power = np.mean(np.abs(rx_sig) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(rx_len) + 1j * np.random.randn(rx_len))
    rx_sig += noise
    return rx_sig


def estimate_delay_doppler_otfs(rx_sig, tx_sig):
    # 理想情况下，我们需要做 OTFS 的解调：先做FFT拆分符号，再做ISFFT回到DD域找峰值
    # 为了演示，这里仍然做一个 brute-force 方式，以对比BPSK，差别仅在发射波形
    max_delay_search = 50
    doppler_search = np.linspace(-400, 400, 81)
    best_metric = -1e12
    est_delay = 0
    est_dopp = 0
    for dly in range(max_delay_search):
        if dly + len(tx_sig) > len(rx_sig):
            break
        segment = rx_sig[dly:dly + len(tx_sig)]
        for df in doppler_search:
            phase = np.exp(-1j * 2 * np.pi * df * np.arange(len(tx_sig)) / Fs)
            # 由于 OTFS 发的是一个“全1”/或Dirac形式的序列，相关性度量类似： sum(segment * conj(phase))
            metric = np.abs(np.sum(segment * np.conj(tx_sig * phase)))
            if metric > best_metric:
                best_metric = metric
                est_delay = dly
                est_dopp = df
    delay_est_s = est_delay / Fs
    doppler_est_hz = est_dopp
    return delay_est_s, doppler_est_hz


delay_rmse_otfs = []
doppler_rmse_otfs = []

for snr in SNR_dBs:
    d_errors = []
    f_errors = []
    for _ in range(N_experiments):
        tx_otfs = generate_otfs_pilot(2000)
        rx_otfs = apply_channel_otfs(tx_otfs, true_delay, true_dopp, snr)
        d_hat, f_hat = estimate_delay_doppler_otfs(rx_otfs, tx_otfs)
        d_errors.append(d_hat - true_delay)
        f_errors.append(f_hat - true_dopp)
    delay_rmse_otfs.append(np.sqrt(np.mean(np.array(d_errors) ** 2)) * 3e8)
    doppler_rmse_otfs.append(np.sqrt(np.mean(np.array(f_errors) ** 2)))

###############################################################################
# 绘制图 1, 2: BPSK vs. OTFS 在无多径(AWGN)下的时延 / 多普勒RMSE
###############################################################################
plt.figure(figsize=(5, 4))
plt.plot(SNR_dBs, delay_rmse_bpsk, 'o-', label="BPSK")
plt.plot(SNR_dBs, delay_rmse_otfs, 's-', label="OTFS")
plt.yscale('log')
plt.grid(True)
plt.xlabel("SNR (dB)")
plt.ylabel("Delay RMSE (meters)")
plt.title("无多径时，时延估计RMSE")
plt.legend()
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
buf.seek(0)
plot1_data = base64.b64encode(buf.read()).decode('ascii')
plt.close()

plt.figure(figsize=(5, 4))
plt.plot(SNR_dBs, doppler_rmse_bpsk, 'o-', label="BPSK")
plt.plot(SNR_dBs, doppler_rmse_otfs, 's-', label="OTFS")
plt.yscale('log')
plt.grid(True)
plt.xlabel("SNR (dB)")
plt.ylabel("Doppler RMSE (Hz)")
plt.title("无多径时，多普勒估计RMSE")
plt.legend()
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
buf.seek(0)
plot2_data = base64.b64encode(buf.read()).decode('ascii')
plt.close()


###############################################################################
# 4. 演示多径场景下的延迟-多普勒谱 & 多径误差包络(MEE)
###############################################################################
#   - 为简化，设置: LOS + 1 条多径（相对时延Delta_d）
#   - 多径相对多普勒Delta_f
###############################################################################

def simulate_delay_doppler_map(tx_sig, delta_dly_samples=20, delta_dopp=50.0, alpha=0.7, snr_db=30):
    # LOS + 1 multipath
    # LOS: delay=0, doppler=0
    # Multipath: delay=delta_dly_samples, doppler=delta_dopp, 幅度alpha
    N_tot = len(tx_sig) + delta_dly_samples
    rx_sig = np.zeros(N_tot, dtype=np.complex64)
    # LOS
    rx_sig[:len(tx_sig)] += tx_sig
    # multipath
    mp_phase = np.exp(1j * 2 * np.pi * delta_dopp * np.arange(len(tx_sig)) / Fs)
    rx_sig[delta_dly_samples:delta_dly_samples + len(tx_sig)] += alpha * tx_sig * mp_phase[:len(tx_sig)]

    # 加噪
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(rx_sig) ** 2)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(N_tot) + 1j * np.random.randn(N_tot))
    rx_sig += noise
    return rx_sig


# 我们对比：BPSK 的模糊函数 & OTFS 的DD域处理
# (下图只是用 brute-force 2D 相关来展示峰值分布)
dly_grid = range(30)
dop_grid = np.linspace(-100, 100, 51)


def ambiguity_function_2d(rx_sig, tx_sig):
    """ 返回一个 [delay_grid, doppler_grid] 的二维相关图。 """
    A = np.zeros((len(dly_grid), len(dop_grid)))
    for i, dly in enumerate(dly_grid):
        if dly + len(tx_sig) > len(rx_sig):
            break
        segment = rx_sig[dly:dly + len(tx_sig)]
        for j, df in enumerate(dop_grid):
            phase = np.exp(-1j * 2 * np.pi * df * np.arange(len(tx_sig)) / Fs)
            A[i, j] = np.abs(np.sum(segment * np.conj(tx_sig * phase)))
    return A


# 生成示例 BPSK 与示例 OTFS 信号
tx_bpsk_ex = generate_bpsk_signal(num_samples=1500)
rx_bpsk_ex = simulate_delay_doppler_map(tx_bpsk_ex, 10, 40.0, alpha=0.6, snr_db=30)
A_bpsk = ambiguity_function_2d(rx_bpsk_ex, tx_bpsk_ex)

tx_otfs_ex = generate_otfs_pilot(1500)
rx_otfs_ex = simulate_delay_doppler_map(tx_otfs_ex, 10, 40.0, alpha=0.6, snr_db=30)
A_otfs = ambiguity_function_2d(rx_otfs_ex, tx_otfs_ex)

# 绘制对比图
plt.figure(figsize=(7, 4))
plt.subplot(1, 2, 1)
plt.imshow(A_bpsk, aspect='auto', extent=[dop_grid[0], dop_grid[-1], dly_grid[-1], dly_grid[0]])
plt.title("BPSK 模糊函数图")
plt.xlabel("Doppler (Hz)")
plt.ylabel("Delay (samples)")

plt.subplot(1, 2, 2)
plt.imshow(A_otfs, aspect='auto', extent=[dop_grid[0], dop_grid[-1], dly_grid[-1], dly_grid[0]])
plt.title("OTFS DD响应图(简化)")
plt.xlabel("Doppler (Hz)")
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
buf.seek(0)
plot3_data = base64.b64encode(buf.read()).decode('ascii')
plt.close()


###############################################################################
# 5. 多径误差包络 (MEE) 示例
###############################################################################
# MEE: 当多径时延相对LOS变化时，测量主峰的偏差
# 在这里演示性地扫描 [0, 100] 个采样点时延差
###############################################################################

def estimate_ranging_bias(rx_sig, tx_sig):
    # 我们只关心LOS主峰位置(或大的峰)，这里简单地找到2D相关最大值对应的 delay
    # 真正的 GNSS/OTFS 定位处理更复杂，这里仅做演示
    A = ambiguity_function_2d(rx_sig, tx_sig)
    idx = np.unravel_index(np.argmax(A), A.shape)
    est_dly = dly_grid[idx[0]]
    return est_dly


delay_candidates = np.arange(0, 101, 10)
mee_bpsk = []
mee_otfs = []
for dly_smpl in delay_candidates:
    # bpsk
    rx_bpsk_mp = simulate_delay_doppler_map(tx_bpsk_ex, dly_smpl, 40, alpha=0.6, snr_db=30)
    est_bpsk_d = estimate_ranging_bias(rx_bpsk_mp, tx_bpsk_ex)
    mee_bpsk.append(est_bpsk_d - 0)  # LOS设定delay=0, multipath为 dly_smpl

    # otfs
    rx_otfs_mp = simulate_delay_doppler_map(tx_otfs_ex, dly_smpl, 40, alpha=0.6, snr_db=30)
    est_otfs_d = estimate_ranging_bias(rx_otfs_mp, tx_otfs_ex)
    mee_otfs.append(est_otfs_d - 0)

plt.figure(figsize=(5, 4))
plt.plot(delay_candidates, mee_bpsk, 'o-', label="BPSK MEE")
plt.plot(delay_candidates, mee_otfs, 's-', label="OTFS MEE")
plt.grid(True)
plt.xlabel("Replica delay w.r.t LOS (samples)")
plt.ylabel("Estimated peak bias (samples)")
plt.title("多径误差包络(MEE)示例(单位:采样点)")
plt.legend()
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
buf.seek(0)
plot4_data = base64.b64encode(buf.read()).decode('ascii')
plt.close()
