import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
from datetime import datetime, timezone
from skyfield.api import load, wgs84

# 从 selection1.py 导入已有的工具函数和 Satellite 类
from selection1 import compute_dgdop, update_satellite_info, Satellite, visable


#############################################
# 1. 基于 TLE 文件的卫星选星环境定义
#############################################
class TLESatelliteSelectionEnvRL:
    """
    基于 TLE 文件的卫星选星环境：
      - 利用 Skyfield 根据 TLE 数据及观测者信息更新卫星的位置信息、仰角、方位角；
      - 根据仰角确定 GEV 特征角 phi，并将方位角转换为弧度作为 beta，构造特征矩阵 [phi, beta, elevation]；
      - 过滤掉低于设定阈值（例如 25°）的卫星；
      - 环境中每次完整决策即生成一个选星序列，组合的 DGDOP 作为 reward（取负值，DGDOP 越低 reward 越高）。
    """

    def __init__(self, tle_file, n_select=4,
                 observer_lat=45.75, observer_lon=126.68, observer_elevation_m=100,
                 obs_time=datetime(2025, 1, 6, 14, 0, 0, tzinfo=timezone.utc),
                 min_elevation=25):
        self.tle_file = tle_file
        self.n_select = n_select
        self.min_elevation = min_elevation

        # 加载 TLE 数据（返回 EarthSatellite 对象列表）
        tle_sats = load.tle_file(self.tle_file)
        print(f"加载了 {len(tle_sats)} 颗卫星 TLE 数据.")
        # 构造 Satellite 对象列表（复用 selection1.py 中已定义的 Satellite 类）
        self.satellites = [Satellite(sat) for sat in tle_sats]

        # 定义观测者（使用 wgs84.topos 构造）
        self.observer = wgs84.latlon(observer_lat, observer_lon, elevation_m=observer_elevation_m)
        # 定义观测时刻
        ts = load.timescale()
        self.t = ts.from_datetime(obs_time)
        # 更新每颗卫星的信息，并过滤掉低于设定仰角的卫星
        valid_sats = []
        for sat in self.satellites:
            update_satellite_info(sat, self.observer, self.t)
            if sat.elevation >= self.min_elevation:
                valid_sats.append(sat)
        self.satellites = valid_sats
        self.num_satellites = len(self.satellites)
        if self.num_satellites == 0:
            raise ValueError("无满足可见条件的卫星，请检查 TLE 文件和观测条件。")
        # 构造特征矩阵：每颗卫星的特征为 [phi, beta, elevation]
        self.feature_dim = 3
        self.features = self._get_features()
        # 转换观测者位置为 ITRS 坐标（km），用于后续 DGDOP 计算
        self.user_pos = self.observer.at(self.t).position.m
        # 初始化 mask（选取时防止重复选择），True 表示卫星可选
        self.mask = np.ones(self.num_satellites, dtype=bool)
        self.selected_idx = []

    def _get_features(self):
        feats = []
        for sat in self.satellites:
            feats.append([sat.phi, sat.beta, sat.elevation])
        return np.array(feats, dtype=np.float32)

    def reset(self):
        """
        重置环境，将 mask 置为全 True，并清空已选索引
        返回：features 和 mask
        """
        self.mask = np.ones(self.num_satellites, dtype=bool)
        self.selected_idx = []
        return self.features, self.mask

    def complete_reward(self, selected_indices):
        """
        根据完整的选星序列 selected_indices 计算组合的 DGDOP，
        并返回 reward（负 DGDOP）
        """
        selected_sats = [self.satellites[i] for i in selected_indices]
        dop = compute_dgdop(selected_sats, self.user_pos)
        reward = -dop
        return reward


#############################################
# 2. PointerNet 模型定义（结合 REINFORCE 策略梯度）
#############################################
class PointerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_select, num_layers=1):
        """
        input_dim: 每颗卫星输入特征的维数（例如 3，对应 [phi, beta, elevation]）
        hidden_dim: LSTM 隐状态维数
        n_select: 需要选择的卫星数量
        num_layers: 编码器层数
        """
        super(PointerNet, self).__init__()
        self.n_select = n_select
        self.hidden_dim = hidden_dim
        # 编码器采用双向 LSTM，输出维数为 2*hidden_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, bidirectional=True)
        # 利用编码器输出的均值初始化解码器状态
        self.init_h = nn.Linear(2 * hidden_dim, hidden_dim)
        self.init_c = nn.Linear(2 * hidden_dim, hidden_dim)
        # 解码器 LSTMCell；输入维度为 2*hidden_dim（采用编码器输出均值作为初始输入）
        self.decoder = nn.LSTMCell(2 * hidden_dim, hidden_dim)
        # Pointer 层：将解码器隐状态映射到与编码器输出相同维度，用于 attention 计算
        self.pointer = nn.Linear(hidden_dim, 2 * hidden_dim)

    def forward(self, inputs):
        """
        inputs: tensor，形状 (batch_size, m, input_dim)，m 为候选卫星数量
        返回：
          selected_indices: tensor，形状 (batch_size, n_select)
          total_log_probs: tensor，形状 (batch_size,)，对应序列的对数概率和
        """
        batch_size, m, _ = inputs.size()
        # 编码器输出 (batch_size, m, 2*hidden_dim)
        encoder_outputs, _ = self.encoder(inputs)
        # 取编码器输出均值作为摘要信息，并初始化解码器状态
        encoder_mean = encoder_outputs.mean(dim=1)  # (batch_size, 2*hidden_dim)
        h = self.init_h(encoder_mean)  # (batch_size, hidden_dim)
        c = self.init_c(encoder_mean)  # (batch_size, hidden_dim)
        # 初始 decoder 输入采用 encoder_mean
        decoder_input = encoder_mean  # (batch_size, 2*hidden_dim)

        selected_indices = []
        log_probs = []
        # 初始化 mask，用于记录哪些卫星已被选中（False 表示未选中）
        mask = torch.zeros(batch_size, m, device=inputs.device, dtype=torch.bool)

        for _ in range(self.n_select):
            h, c = self.decoder(decoder_input, (h, c))  # h: (batch_size, hidden_dim)
            # 将 decoder 隐状态映射为查询向量
            query = self.pointer(h)  # (batch_size, 2*hidden_dim)
            # 计算注意力得分：采用点积计算
            scores = torch.bmm(encoder_outputs, query.unsqueeze(2)).squeeze(2)  # (batch_size, m)
            # 屏蔽已选择的卫星（inplace 修改前先 clone）
            mask = mask.clone()
            scores = scores.masked_fill(mask, -1e9)
            probs = torch.softmax(scores, dim=1)
            dist = D.Categorical(probs)
            selected = dist.sample()  # (batch_size,)
            log_prob = dist.log_prob(selected)
            selected_indices.append(selected)
            log_probs.append(log_prob)
            # 更新 mask：选中卫星置为 True
            mask = mask.clone()
            mask[torch.arange(batch_size), selected] = True
            # 更新 decoder 输入为当前选中卫星的 encoder 输出
            decoder_input = encoder_outputs[torch.arange(batch_size), selected]

        selected_indices = torch.stack(selected_indices, dim=1)  # (batch_size, n_select)
        log_probs = torch.stack(log_probs, dim=1)  # (batch_size, n_select)
        total_log_probs = log_probs.sum(dim=1)  # (batch_size,)
        return selected_indices, total_log_probs


#############################################
# 3. REINFORCE训练函数（基于 PointerNet）
#############################################
def train_pointer_net_env(model, optimizer, env, num_episodes=500, device="cpu"):
    """
    model: PointerNet 模型
    optimizer: 优化器
    env: TLESatelliteSelectionEnvRL 环境
    num_episodes: 训练轮数（episode 数）
    device: 设备
    """
    model.train()
    all_rewards = []

    for episode in range(num_episodes):
        # 重置环境，获得卫星特征矩阵和 mask（环境中 mask 用于滤除低仰角卫星，但本处输入仅用特征）
        features_np, _ = env.reset()
        # 将 features 转为 tensor，shape: (1, m, feature_dim)
        input_tensor = torch.tensor(features_np, dtype=torch.float, device=device).unsqueeze(0)
        # 前向传播：获得选取的卫星索引序列和对应 log 概率和
        selected_indices, log_prob = model(input_tensor)
        # selected_indices: (1, n_select)，转换为 list
        indices = selected_indices.squeeze(0).tolist()
        # 根据所选卫星索引计算 reward（负 DGDOP）
        reward = env.complete_reward(indices)
        all_rewards.append(reward)
        # REINFORCE 损失：loss = - log_prob * reward
        loss = - log_prob * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            print(
                f"Episode {episode + 1}: Loss = {loss.item():.4f}, Reward = {reward:.4f}, Avg Reward = {avg_reward:.4f}")

    return model, all_rewards


#############################################
# 4. 主程序：加载 TLE 数据，训练 PointerNet，在线测试
#############################################
if __name__ == "__main__":
    # 为保证可重复性设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)

    # 指定 TLE 文件路径（请根据实际情况修改）
    tle_file = "satellite_data/starlink_tle_20250106_144506.txt"

    # 环境参数
    n_select = 4
    observer_lat = 45.75
    observer_lon = 126.68
    observer_elevation_m = 100
    env = TLESatelliteSelectionEnvRL(tle_file=tle_file, n_select=n_select,
                                     observer_lat=observer_lat,
                                     observer_lon=observer_lon,
                                     observer_elevation_m=observer_elevation_m)

    # 构造 PointerNet 模型，输入特征维数为 3（phi, beta, elevation）
    feature_dim = 3
    hidden_dim = 128
    model = PointerNet(input_dim=feature_dim, hidden_dim=hidden_dim, n_select=n_select)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 开始训练 REINFORCE 模型
    trained_model, rewards_history = train_pointer_net_env(model, optimizer, env, num_episodes=500, device=device)

    # 在线测试：利用训练后的 PointerNet 模型生成选星序列
    trained_model.eval()
    with torch.no_grad():
        features_np, _ = env.reset()
        input_tensor = torch.tensor(features_np, dtype=torch.float, device=device).unsqueeze(0)
        selected_indices, _ = trained_model(input_tensor)
        selected_indices = selected_indices.squeeze(0).tolist()
        print("模型选取的卫星索引：", selected_indices)

        # 根据所选卫星计算组合的 DGDOP
        final_reward = env.complete_reward(selected_indices)
        print("组合对应的 DGDOP（负 reward）：", -final_reward)