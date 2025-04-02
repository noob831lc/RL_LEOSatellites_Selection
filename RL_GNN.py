import time
from datetime import timezone, datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt  # 用于绘图

# Skyfield 相关导入
from skyfield.api import load
from skyfield.toposlib import wgs84


# ------------------------------
# 1. 计算 DGDOP 函数
# ------------------------------
def compute_dgdop_combo(selected_positions, user_pos):
    """
    selected_positions: list，每个元素为一个卫星的 ECEF 坐标 (np.array, shape (3,))
    user_pos: np.array，观测者（用户）在 ITRS 坐标系下的位置（单位：m）
    返回：DGDOP 值（数值越小表示定位几何越优）
    """
    G = []
    for pos in selected_positions:
        diff = pos - user_pos
        norm_diff = np.linalg.norm(diff)
        if norm_diff == 0:
            continue
        unit_vector = diff / norm_diff
        row = np.hstack([unit_vector, [1]])
        G.append(row)
    G = np.array(G)
    epsilon = 1e-8
    try:
        Q = np.linalg.inv(G.T @ G + epsilon * np.eye(G.shape[1]))
        dop = np.sqrt(abs(np.trace(Q)))
    except np.linalg.LinAlgError:
        dop = np.inf
    return 10 * dop


# ------------------------------
# 2. 定义图卷积层及 GNN 模块
# ------------------------------
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        # x: (num_nodes, in_features)
        # adj: (num_nodes, num_nodes) 邻接矩阵
        support = self.linear(x)
        out = torch.matmul(adj, support)
        return out


class GNNBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        """
        两层图卷积网络作为 GNN 模块
        """
        super(GNNBlock, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden_features)
        self.gc2 = GraphConvolution(hidden_features, out_features)

    def forward(self, x, adj):
        x = torch.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x


# ------------------------------
# 3. 定义结合 GNN 的 Pointer Network 模型
# ------------------------------
class PointerNet_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_select, num_layers=1, gnn_hidden_dim=None, gnn_out_dim=None):
        """
        input_dim: 原始候选卫星特征维度
        hidden_dim: LSTM 隐藏层单元数
        n_select: 需要选择的卫星数量
        gnn_hidden_dim, gnn_out_dim: 分别为 GNN 隐层和输出层维度；若不指定默认为 hidden_dim 和 input_dim
        """
        super(PointerNet_GNN, self).__init__()
        self.n_select = n_select
        if gnn_hidden_dim is None:
            gnn_hidden_dim = hidden_dim
        if gnn_out_dim is None:
            gnn_out_dim = input_dim
        # GNN 模块，将原始特征融合图结构信息
        self.gnn = GNNBlock(input_dim, gnn_hidden_dim, gnn_out_dim)
        # 双向 LSTM 编码器
        self.encoder = nn.LSTM(gnn_out_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, bidirectional=True)
        self.init_h = nn.Linear(2 * hidden_dim, hidden_dim)
        self.init_c = nn.Linear(2 * hidden_dim, hidden_dim)
        # LSTMCell 作为解码器
        self.decoder = nn.LSTMCell(2 * hidden_dim, hidden_dim)
        self.pointer = nn.Linear(hidden_dim, 2 * hidden_dim)

    def forward(self, inputs, adj):
        """
        inputs: tensor, 形状 (batch_size, m, input_dim)，其中 m 表示候选卫星数量
        adj: tensor, 形状 (batch_size, m, m)，每个样本的邻接矩阵
        返回：
          selected_indices: tensor, 形状 (batch_size, n_select) ，表示选出的卫星索引序列
          total_log_probs: tensor, 形状 (batch_size, ) ，该序列的对数概率和
        """
        batch_size, m, _ = inputs.size()
        # 对每个样本使用 GNN 增强特征
        gnn_out = []
        for i in range(batch_size):
            xi = inputs[i]  # (m, input_dim)
            adji = adj[i]  # (m, m)
            gnn_feature = self.gnn(xi, adji)  # (m, gnn_out_dim)
            gnn_out.append(gnn_feature)
        gnn_out = torch.stack(gnn_out, dim=0)  # (batch_size, m, gnn_out_dim)

        encoder_outputs, _ = self.encoder(gnn_out)  # (batch_size, m, 2*hidden_dim)
        encoder_mean = encoder_outputs.mean(dim=1)  # (batch_size, 2*hidden_dim)
        h = self.init_h(encoder_mean)  # (batch_size, hidden_dim)
        c = self.init_c(encoder_mean)  # (batch_size, hidden_dim)
        decoder_input = encoder_mean

        selected_indices = []
        log_probs = []
        mask = torch.zeros(batch_size, m, device=inputs.device, dtype=torch.bool)

        for _ in range(self.n_select):
            h, c = self.decoder(decoder_input, (h, c))  # (batch_size, hidden_dim)
            query = self.pointer(h)  # (batch_size, 2*hidden_dim)
            scores = torch.bmm(encoder_outputs, query.unsqueeze(2)).squeeze(2)  # (batch_size, m)
            scores = scores.masked_fill(mask, -1e9)
            probs = torch.softmax(scores, dim=1)  # (batch_size, m)
            dist = torch.distributions.Categorical(probs)
            selected = dist.sample()  # (batch_size,)
            log_prob = dist.log_prob(selected)
            selected_indices.append(selected)
            log_probs.append(log_prob)
            mask = mask.clone()
            mask[torch.arange(batch_size), selected] = True
            decoder_input = encoder_outputs[torch.arange(batch_size), selected]

        selected_indices = torch.stack(selected_indices, dim=1)  # (batch_size, n_select)
        log_probs = torch.stack(log_probs, dim=1)  # (batch_size, n_select)
        total_log_probs = log_probs.sum(dim=1)  # (batch_size,)
        return selected_indices, total_log_probs


# ------------------------------
# 4. 构造邻接矩阵函数
# ------------------------------
def compute_adjacency(candidate_positions, threshold=1000.0):
    """
    candidate_positions: numpy 数组，形状 (m, 3)，单位为米
    threshold: 距离阈值，当两颗卫星之间的距离小于该阈值时视为相邻
    返回：归一化后的邻接矩阵，形状 (m, m)
    """
    m = candidate_positions.shape[0]
    adj = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            if i == j:
                adj[i, j] = 1.0
            else:
                dist = np.linalg.norm(candidate_positions[i] - candidate_positions[j])
                if dist < threshold:
                    adj[i, j] = 1.0
    # 简单归一化（按每行除以行和）
    deg = np.sum(adj, axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    adj = adj / deg
    return adj


# ------------------------------
# 5. 训练过程（REINFORCE 策略）
# ------------------------------
def train_pointer_net(model, optimizer, epochs, batch_size, candidate_features, candidate_positions, user_pos):
    """
    model: PointerNet_GNN 模型
    optimizer: PyTorch 优化器
    epochs: 训练轮数
    batch_size: 批大小
    candidate_features: tensor，形状 (m, feature_dim)，候选卫星特征
    candidate_positions: numpy 数组，形状 (m, 3)，候选卫星的 ECEF 坐标
    user_pos: numpy 数组，观测者在 ITRS 坐标系下的位置（单位：m）
    """
    device = next(model.parameters()).device
    m, feature_dim = candidate_features.size()
    # 构造邻接矩阵（采用相同阈值，每个样本共享同一图结构）
    adj = compute_adjacency(candidate_positions, threshold=1000.0)
    adj = torch.tensor(adj, dtype=torch.float, device=device)
    adj = adj.unsqueeze(0).repeat(batch_size, 1, 1)

    reward_history = []
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        # 扩展 candidate_features 至 batch 维度：(batch_size, m, feature_dim)
        inputs = candidate_features.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        selected_indices, log_probs = model(inputs, adj)

        # 计算每个样本的奖励（使用 DGDOP 作为指标：reward = -DGDOP）
        rewards = []
        selected_indices_np = selected_indices.cpu().detach().numpy()
        for i in range(batch_size):
            indices = selected_indices_np[i].tolist()
            sel_positions = [candidate_positions[j] for j in indices]
            dop = compute_dgdop_combo(sel_positions, user_pos)
            reward = 1 / (1 + dop)  # DGDOP 越低，reward 越高
            rewards.append(reward)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)

        loss = - (log_probs * rewards).mean()
        loss.backward()
        optimizer.step()

        mean_reward = rewards.mean().item()
        loss_val = loss.item()
        reward_history.append(mean_reward)
        loss_history.append(loss_val)

        print(f"Epoch {epoch}: Loss = {loss_val:.4f}, Mean Reward = {mean_reward:.4f}")

    # 绘制训练曲线
    window = 100  # 设置窗口大小，根据数据量调整最佳取值
    smooth_loss = np.convolve(loss_history, np.ones(window) / window, mode='valid')
    smooth_reward = np.convolve(reward_history, np.ones(window) / window, mode='valid')
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(smooth_loss, label='Smoothed Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(smooth_reward, label='Smoothed Mean Reward', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Reward')
    plt.title('Training Reward Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model


# ------------------------------
# 6. 主函数示例
# ------------------------------
if __name__ == '__main__':
    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    ts = load.timescale()
    tle_file = "satellite_data/starlink_tle_20250106_144506.txt"
    t = ts.from_datetime(datetime(2025, 1, 6, 14, 0, 0, tzinfo=timezone.utc))
    satellites = load.tle_file(tle_file)
    print(f"加载了 {len(satellites)} 颗卫星 TLE 数据.")

    # 定义观测者（以 wgs84.topos 方式定义）
    observer = wgs84.latlon(45.75, 126.68, elevation_m=100)
    user_pos = np.array(observer.at(t).position.m)  # 观测者位置，单位：米


    # 过滤出可见卫星（例如：仰角 > 25°）
    def visable(sat, observer, t):
        difference = sat - observer
        topocentric = difference.at(t)
        alt, az, _ = topocentric.altaz()
        return alt.degrees > 25


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
    trained_model = train_pointer_net(model, optimizer, epochs, batch_size,
                                      candidate_features, candidate_positions, user_pos)
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
        dop = compute_dgdop_combo(sel_positions, user_pos)
        print(f"RL-GNN: DGDOP = {dop:.2f}, Time = {elapsed:.4f} ms")
