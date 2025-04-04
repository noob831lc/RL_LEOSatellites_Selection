import torch
import torch.optim as optim
import time
import numpy as np
from datetime import datetime, timezone
from utils.leoparam import observer, visable
from utils.tle import parse_tle_data
from utils.tle import timescale as ts
from RL_model.model import PointerNet_GNN, compute_adjacency
from RL_model.train import train
from algorithm.Selection import compute_dgdop

if __name__ == '__main__':
    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    tle_file = "satellite_data/starlink_tle_20250106_144506.txt"
    t = ts.from_datetime(datetime(2025, 1, 5, 14, 0, 0, tzinfo=timezone.utc))
    satellites = parse_tle_data(tle_file)
    print(f"加载了 {len(satellites)} 颗卫星 TLE 数据.")

    # 定义观测者（以 wgs84.topos 方式定义）
    ground_station = observer(45.75, 126.68, elevation_m=100)
    user_pos = np.array(ground_station.at(t).position.m)  # 观测者位置，单位：米

    vis_sats = [sat for sat in satellites if visable(sat, observer, t)]
    m = len(vis_sats)
    print(f"可见卫星数量: {m}")

    # 构造候选卫星的特征和位置
    # 这里采用的特征为 [phi, beta]
    feature_dim = 2
    candidate_positions = np.empty((m, 3))
    candidate_features = np.empty((m, feature_dim))
    for i, sat in enumerate(vis_sats):
        candidate_positions[i] = sat.at(t).position.m
        difference = sat - observer
        topocentric = difference.at(t)
        alt, az, _ = topocentric.altaz()
        elevation = alt.degrees
        # 根据仰角定义 phi
        phi = (2 * np.pi / 3) if elevation >= 50 else (np.pi / 3)
        # beta 为方位角（转换为弧度）
        beta = np.radians(az.degrees)
        candidate_features[i, 0] = phi
        candidate_features[i, 1] = beta

    candidate_features = torch.tensor(candidate_features, dtype=torch.float)

    # 初始化 PointerNet_GNN 模型
    n_select = 4
    hidden_dim = 128
    model = PointerNet_GNN(input_dim=feature_dim, hidden_dim=hidden_dim, n_select=n_select)
    print("使用设备:", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    epochs = 30000
    batch_size = 16

    # 训练模型并绘制训练曲线
    trained_model = train(model, optimizer, epochs, 
                          batch_size,candidate_features, 
                          candidate_positions, user_pos)
    #
    # 保存训练后的模型
    model_save_path = 'pointer_net_gnn.pth'
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"训练后的模型已保存到 {model_save_path}")

    # 加载保存的模型进行测试
    test_model = PointerNet_GNN(input_dim=feature_dim, hidden_dim=hidden_dim, n_select=n_select)
    test_model.to(device)
    test_model.load_state_dict(torch.load(model_save_path,weights_only=True, map_location=device))
    test_model.eval()

    # ------------------------------
    # 调整部分：热身运行与同步 GPU
    # ------------------------------
    with torch.no_grad():
        inputs = candidate_features.unsqueeze(0).to(device)  # (1, m, feature_dim)
        adj = compute_adjacency(candidate_positions, threshold=1000.0)
        adj = torch.tensor(adj, dtype=torch.float, device=device).unsqueeze(0)  # (1, m, m)
        # 热身运行一次前向传播，让 GPU 完成初始化
        _ = test_model(inputs, adj)
        if device.type == 'cuda':
            torch.cuda.synchronize()  # 同步 GPU 操作

    # ------------------------------
    # 开始正式的计时测试
    # ------------------------------
    with torch.no_grad():
        inputs = candidate_features.unsqueeze(0).to(device)  # (1, m, feature_dim)
        adj = compute_adjacency(candidate_positions, threshold=1000.0)
        adj = torch.tensor(adj, dtype=torch.float, device=device).unsqueeze(0)  # (1, m, m)
        if device.type == 'cuda':
            torch.cuda.synchronize()  # 在计时前同步 GPU
        start = time.perf_counter()  # 开始计时
        selected_indices, _ = test_model(inputs, adj)
        if device.type == 'cuda':
            torch.cuda.synchronize()  # 确保模型推理已完成之后再计时
        elapsed = (time.perf_counter() - start) * 1000  # 毫秒计时
        selected_indices = selected_indices.cpu().numpy().tolist()[0]
        selected_names = [vis_sats[j].name for j in selected_indices]
        print("加载模型后，模型选择的卫星组合:", selected_names)
        sel_positions = [candidate_positions[j] for j in selected_indices]
        dop = compute_dgdop(sel_positions, user_pos)
        print(f"RL-GNN: DGDOP = {dop:.2f}, Time = {elapsed:.4f} ms")