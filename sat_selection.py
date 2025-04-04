import time
import os
import matplotlib.pyplot as plt
from pylab import mpl

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
mpl.rcParams["axes.unicode_minus"] = False


if __name__ == '__main__':
    pass
    # ------------------------------
    # 基础设置与 TLE 数据加载
    # ------------------------------
    # ts = load.timescale()
    # t = ts.from_datetime(datetime(2025, 1, 6, 14, 0, 0, tzinfo=timezone.utc))
    # tle_file = 'satellite_data/starlink_tle_20250106_144506.txt'
    # satellites_tle = load.tle_file(tle_file)
    # print(f"加载了 {len(satellites_tle)} 颗卫星 TLE 数据.")

    # # 定义观测者
    # observer = wgs84.latlon(45.75, 126.68, elevation_m=100)
    # user_pos = observer.at(t).position.m  # 单位 m

    # # ------------------------------
    # # Selection1 部分：转换为 Satellite 对象并更新信息
    # # ------------------------------
    # satellites_sel = [Satellite(sat) for sat in satellites_tle]
    # satellites_sel = [sat for sat in satellites_sel if visable(sat, observer, t)]
    # print(f"[Selection1] 可见卫星数量: {len(satellites_sel)}")
    # for sat in satellites_sel:
    #     update_satellite_info(sat, observer, t)

    # n = 4  # 选择卫星数量

    # # OSSA 算法
    # start = time.perf_counter()
    # ossa_combo, ossa_dop = ossa_selection(satellites_sel, user_pos, n)
    # elapsed_ossa = (time.perf_counter() - start) * 1000
    # ossa_names = [sat.name for sat in ossa_combo]
    # print("\n[OSSA] 卫星组合:", ossa_names)
    # print(f"[OSSA] DGDOP = {ossa_dop:.2f}, 时间 = {elapsed_ossa:.4f} ms")

    # # MaxEle 算法
    # start = time.perf_counter()
    # maxele_combo, maxele_dop = maxele_selection(satellites_sel, user_pos, n)
    # elapsed_maxele = (time.perf_counter() - start) * 1000
    # maxele_names = [sat.name for sat in maxele_combo]
    # print("\n[MaxEle] 卫星组合:", maxele_names)
    # print(f"[MaxEle] DGDOP = {maxele_dop:.2f}, 时间 = {elapsed_maxele:.4f} ms")

    # # FCSDp 算法
    # start = time.perf_counter()
    # fcsdp_combo, fcsdp_dop = fcsdp_selection(satellites_sel, user_pos, n)
    # elapsed_fcsdp = (time.perf_counter() - start) * 1000
    # fcsdp_names = [sat.name for sat in fcsdp_combo]
    # print("\n[FCSDp] 卫星组合:", fcsdp_names)
    # print(f"[FCSDp] DGDOP = {fcsdp_dop:.2f}, 时间 = {elapsed_fcsdp:.4f} ms")

    # # ------------------------------
    # # RL_GNN 测试部分：加载训练好的模型并进行推理
    # # ------------------------------
    # # 从 TLE 数据中过滤出满足仰角>25°的 EarthSatellite 对象（用于 RL_GNN 测试）
    # rl_satellites = [sat for sat in satellites_tle if visable_rl(sat, observer, t)]
    # m_rl = len(rl_satellites)

    # # 构造候选卫星的特征与位置（单位均为 m 或弧度）
    # feature_dim = 2
    # candidate_positions = np.empty((m_rl, 3))
    # candidate_features = np.empty((m_rl, feature_dim))
    # for i, sat in enumerate(rl_satellites):
    #     candidate_positions[i] = sat.at(t).position.m
    #     diff = sat - observer
    #     topocentric = diff.at(t)
    #     alt, az, _ = topocentric.altaz()
    #     elevation = alt.degrees
    #     phi = 2 * np.pi / 3 if elevation >= 50 else np.pi / 3
    #     beta = np.radians(az.degrees)
    #     candidate_features[i, 0] = phi
    #     candidate_features[i, 1] = beta

    # candidate_features = torch.tensor(candidate_features, dtype=torch.float)

    # # 加载训练好的 RL_GNN 模型
    # n_select = 4
    # hidden_dim = 128
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # rl_model = PointerNet_GNN(input_dim=feature_dim, hidden_dim=hidden_dim, n_select=n_select)
    # rl_model.to(device)
    # model_save_path = 'pointer_net_gnn.pth'
    # rl_model.load_state_dict(torch.load(model_save_path,weights_only=True, map_location=device))
    # rl_model.eval()

    # # 前向推理前热身并同步 GPU
    # with torch.no_grad():
    #     inputs = candidate_features.unsqueeze(0).to(device)
    #     adj = compute_adjacency(candidate_positions, threshold=1000.0)
    #     adj = torch.tensor(adj, dtype=torch.float, device=device).unsqueeze(0)
    #     _ = rl_model(inputs, adj)
    #     if device.type == 'cuda':
    #         torch.cuda.synchronize()

    # with torch.no_grad():
    #     inputs = candidate_features.unsqueeze(0).to(device)
    #     adj = compute_adjacency(candidate_positions, threshold=1000.0)
    #     adj = torch.tensor(adj, dtype=torch.float, device=device).unsqueeze(0)
    #     if device.type == 'cuda':
    #         torch.cuda.synchronize()
    #     start = time.perf_counter()
    #     selected_indices, _ = rl_model(inputs, adj)
    #     if device.type == 'cuda':
    #         torch.cuda.synchronize()
    #     elapsed_rl = (time.perf_counter() - start) * 1000  # ms
    #     selected_indices = selected_indices.cpu().numpy().tolist()[0]
    #     rl_names = [rl_satellites[j].name for j in selected_indices]
    #     sel_positions_rl = [candidate_positions[j] for j in selected_indices]
    #     rl_dop = compute_dgdop_combo(sel_positions_rl, user_pos)

    # print("\n[RL-GNN] 卫星组合:", rl_names)
    # print(f"[RL-GNN] DGDOP = {rl_dop:.2f}, 时间 = {elapsed_rl:.4f} ms")

    # # ------------------------------
    # # 综合对比结果与绘图
    # # ------------------------------
    # method_names = ['RL-GNN', 'OSSA', 'FCSDp', 'MaxEle']
    # dop_values = [rl_dop, ossa_dop, fcsdp_dop, maxele_dop]
    # time_values = [elapsed_rl, elapsed_ossa, elapsed_fcsdp, elapsed_maxele]

    # print("\n===== 综合对比结果 =====")
    # for i, method in enumerate(method_names):
    #     if method == 'RL-GNN':
    #         names = rl_names
    #     elif method == 'OSSA':
    #         names = ossa_names
    #     elif method == 'FCSDp':
    #         names = fcsdp_names
    #     else:
    #         names = maxele_names
    #     print(f"{method}: 组合 = {names}, DGDOP = {dop_values[i]:.2f}, 时间 = {time_values[i]:.4f} ms")

    # x = np.arange(len(method_names))
    # width = 0.2

    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # # DGDOP 对比
    # axes[0].bar(x, dop_values, width, color=['skyblue', 'salmon', 'lightgreen', 'plum'],label=method_names)
    # axes[0].set_xticks(x)
    # axes[0].set_xticklabels(method_names)
    # axes[0].set_ylabel('DGDOP')
    # axes[0].set_title('DGDOP')
    # axes[0].legend()

    # # 执行时间对比
    # axes[1].set_yscale('log')
    # axes[1].bar(x, time_values, width, color=['skyblue', 'salmon', 'lightgreen', 'plum'],label=method_names)
    # axes[1].set_xticks(x)
    # axes[1].set_xticklabels(method_names)
    # axes[1].set_ylabel('time (ms)')
    # axes[1].set_title('time')
    # axes[1].legend()

    # plt.tight_layout()
    # plt.show()