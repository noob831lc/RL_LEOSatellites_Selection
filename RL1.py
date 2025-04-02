import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
from datetime import datetime, timezone

# 导入 Skyfield 模块
from skyfield.api import load, wgs84
# 从 selection1.py 导入已有的工具函数（例如 compute_dgdop、update_satellite_info 和 visable）
from selection1 import compute_dgdop, update_satellite_info, visable
from selection1 import Satellite


#############################################
# 1. 基于 TLE 文件的环境定义
#############################################
class TLESatelliteSelectionEnvRL:
    """
    基于 TLE 文件的卫星选星环境：
      - 通过 TLE 文件加载实际卫星数据；
      - 依据观测者位置与给定观测时刻更新每颗卫星的位置信息、仰角、方位角，
        并由仰角确定 GEV 特征角 phi，同时将方位角转换为弧度得到 beta；
      - 过滤掉低于设定仰角阈值的卫星（例如 25°），只保留可见卫星；
      - 构造特征矩阵：每颗卫星的特征定义为 [phi, beta, elevation]；
      - 每一步选取一颗卫星，当累计选取数量达到 n_select 后，计算组合的 DGDOP，
        并以负 DGDOP 作为 reward 返回。
    """

    def __init__(self, tle_file, n_select=4,
                 observer_lat=15.75, observer_lon=126.68, observer_elevation_m=100,
                 obs_time=datetime(2025, 1, 6, 14, 0, 0, tzinfo=timezone.utc),
                 min_elevation=25):
        self.tle_file = tle_file
        self.n_select = n_select
        self.min_elevation = min_elevation
        # 加载 TLE 数据（返回 EarthSatellite 对象列表）
        tle_sats = load.tle_file(self.tle_file)
        print(f"加载了 {len(tle_sats)} 颗卫星 TLE 数据.")
        # 使用已有的 Satellite 类构造卫星对象
        # 这里直接复用 selection1.py 中的 Satellite 类定义

        self.satellites = [Satellite(sat) for sat in tle_sats]

        # 定义观测者：采用 Skyfield 的 wgs84.topos
        self.observer = wgs84.latlon(observer_lat, observer_lon, elevation_m=observer_elevation_m)
        # 定义观测时刻
        ts = load.timescale()
        self.t = ts.from_datetime(obs_time)
        # 更新每颗卫星的信息：计算其 ECEF 坐标、仰角、方位角、phi、beta 等
        valid_sats = []
        for sat in self.satellites:
            update_satellite_info(sat, self.observer, self.t)
            # 过滤掉低于设定仰角的卫星
            if sat.elevation >= self.min_elevation:
                valid_sats.append(sat)
        self.satellites = valid_sats
        self.num_satellites = len(self.satellites)
        if self.num_satellites == 0:
            raise ValueError("无满足可见条件的卫星，请检查 TLE 文件和观测条件。")
        # 构造特征矩阵，顺序：[phi, beta, elevation]
        self.feature_dim = 3
        self.features = self._get_features()
        # 观测者位置，用于后续 DGDOP 计算，转换成 ITRS 坐标（km）
        self.user_pos = self.observer.at(self.t).position.m
        # 初始化 mask：True 表示该卫星可选
        self.mask = np.ones(self.num_satellites, dtype=bool)
        self.selected_idx = []

    def _get_features(self):
        feats = []
        for sat in self.satellites:
            feats.append([sat.phi, sat.beta, sat.elevation])
        return np.array(feats, dtype=np.float32)

    def reset(self):
        """
        重置环境，将 mask 重置为全 True，清空已选索引列表
        """
        self.mask = np.ones(self.num_satellites, dtype=bool)
        self.selected_idx = []
        return self.features, self.mask

    def step(self, action):
        """
        执行动作：选取某颗卫星（动作为卫星在当前特征矩阵中的索引）
        当累计选取达到 n_select 时，计算组合的 DGDOP，
        并以负 DGDOP 作为 reward 返回。
        """
        if not self.mask[action]:
            raise ValueError("所选卫星已被选取！")
        self.selected_idx.append(action)
        self.mask[action] = False
        done = (len(self.selected_idx) >= self.n_select)
        reward = 0.0
        if done:
            selected_sats = [self.satellites[i] for i in self.selected_idx]
            dop = compute_dgdop(selected_sats, self.user_pos)
            reward = -dop  # 负 DGDOP 越高代表组合性能越好
        return (self.features, self.mask), reward, done, {}


#############################################
# 2. PPO 网络及辅助函数（保持不变）
#############################################
def masked_logits(logits, mask_tensor):
    neg_inf = -1e8
    return logits.masked_fill(~mask_tensor, neg_inf)


class PPONetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super(PPONetwork, self).__init__()
        # Actor 网络：对每颗卫星的特征输出一个得分
        self.actor_fc1 = nn.Linear(feature_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, 1)
        # Critic 网络：采用所有未选卫星的特征加权平均描述全局状态
        self.critic_fc1 = nn.Linear(feature_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, 1)

    def forward_actor(self, features):
        x = torch.relu(self.actor_fc1(features))
        logits = self.actor_fc2(x).squeeze(-1)
        return logits

    def forward_critic(self, features, mask):
        mask = mask.unsqueeze(1)
        mask_float = mask.float()
        if mask_float.sum() > 0:
            x = (features * mask_float).sum(dim=0, keepdim=True) / mask_float.sum()
        else:
            x = features.mean(dim=0, keepdim=True)
        x = torch.relu(self.critic_fc1(x))
        value = self.critic_fc2(x)
        return value.squeeze(0)


#############################################
# 3. PPO 算法的训练实现（保持不变）
#############################################
def ppo_update(network, optimizer, batch_states, batch_masks, batch_actions, batch_logprobs, batch_returns,
               batch_advantages,
               clip_epsilon=0.2, critic_coef=0.5, entropy_coef=0.01, epochs=4, mini_batch_size=32):
    batch_size = len(batch_states)
    indices = np.arange(batch_size)
    for epoch in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mb_idx = indices[start:end]
            states_mb = [batch_states[i] for i in mb_idx]  # 每项为 (features, mask)
            features_mb = torch.stack([torch.tensor(s[0], dtype=torch.float) for s in states_mb])
            masks_mb = torch.stack([torch.tensor(s[1], dtype=torch.bool) for s in states_mb])
            actions_mb = torch.tensor([batch_actions[i] for i in mb_idx])
            old_logprobs_mb = torch.tensor([batch_logprobs[i] for i in mb_idx], dtype=torch.float)
            returns_mb = torch.tensor([batch_returns[i] for i in mb_idx], dtype=torch.float)
            advantages_mb = torch.tensor([batch_advantages[i] for i in mb_idx], dtype=torch.float)
            new_logprobs = []
            values = []
            entropies = []
            for i in range(features_mb.shape[0]):
                feats = features_mb[i]
                mask = masks_mb[i]
                logits = network.forward_actor(feats)
                logits_masked = masked_logits(logits, mask)
                dist = D.Categorical(logits=logits_masked)
                new_logprob = dist.log_prob(actions_mb[i])
                entropy = dist.entropy()
                new_logprobs.append(new_logprob)
                value = network.forward_critic(feats, mask)
                values.append(value)
                entropies.append(entropy)
            new_logprobs = torch.stack(new_logprobs)
            values = torch.stack(values)
            entropies = torch.stack(entropies)
            ratio = torch.exp(new_logprobs - old_logprobs_mb)
            surr1 = ratio * advantages_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = torch.nn.functional.mse_loss(values.squeeze(-1), returns_mb)
            entropy_loss = -entropies.mean()
            loss = policy_loss + critic_coef * value_loss + entropy_coef * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_ppo(agent, env, num_epochs=5000, batch_size=16, clip_epsilon=0.2):
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    gamma = 0.99
    all_episode_rewards = []
    batch_states = []
    batch_actions = []
    batch_logprobs = []
    batch_returns = []
    batch_advantages = []
    episode_count = 0

    while episode_count < num_epochs:
        states = []
        actions = []
        logprobs = []
        rewards = []
        values = []

        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            features_np, mask_np = obs
            features_tensor = torch.tensor(features_np, dtype=torch.float)
            mask_tensor = torch.tensor(mask_np, dtype=torch.bool)
            logits = agent.forward_actor(features_tensor)
            logits_masked = masked_logits(logits, mask_tensor)
            dist = D.Categorical(logits=logits_masked)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = agent.forward_critic(features_tensor, mask_tensor)
            states.append(obs)
            actions.append(action.item())
            logprobs.append(log_prob.item())
            values.append(value.item())
            obs, reward, done, _ = env.step(action.item())
            rewards.append(reward)
            episode_reward += reward

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        values = np.array(values)
        advantages = returns - values

        batch_states.extend(states)
        batch_actions.extend(actions)
        batch_logprobs.extend(logprobs)
        batch_returns.extend(returns)
        batch_advantages.extend(advantages)

        episode_count += 1
        all_episode_rewards.append(episode_reward)

        if episode_count % batch_size == 0:
            ppo_update(agent, optimizer, batch_states,
                       [s[1] for s in batch_states],
                       batch_actions, batch_logprobs, batch_returns, batch_advantages,
                       clip_epsilon=clip_epsilon)
            batch_states = []
            batch_actions = []
            batch_logprobs = []
            batch_returns = []
            batch_advantages = []
            avg_reward = np.mean(all_episode_rewards[-batch_size:])
            print(f"Episode: {episode_count}, Average Reward: {avg_reward:.4f}")

    return agent, all_episode_rewards


#############################################
# 4. 主程序：训练和测试基于 TLE 数据的 PPO 选星算法
#############################################
if __name__ == "__main__":
    # 设置随机种子，保证实验可重复
    np.random.seed(42)
    torch.manual_seed(42)

    # 指定 TLE 文件路径（请替换为你本地的 TLE 文件路径）
    tle_file = "satellite_data/starlink_tle_20250106_144506.txt"

    # 定义环境参数，利用实际 TLE 数据
    n_select = 4
    # 观测者位置（经纬度及海拔均可根据实际情况设置）
    observer_lat = 45.75
    observer_lon = 126.68
    observer_elevation_m = 100
    env = TLESatelliteSelectionEnvRL(tle_file=tle_file, n_select=n_select,
                                     observer_lat=observer_lat, observer_lon=observer_lon,
                                     observer_elevation_m=observer_elevation_m)

    # 定义智能体（输入特征维数为 3）
    feature_dim = 3
    agent = PPONetwork(feature_dim, hidden_dim=64)

    # 离线训练 PPO 模型
    trained_agent, rewards_history = train_ppo(agent, env, num_epochs=50000, batch_size=16, clip_epsilon=0.2)

    # 在线测试：利用训练后的模型选取卫星
    obs = env.reset()
    done = False
    selected_actions = []
    while not done:
        features_np, mask_np = obs
        features_tensor = torch.tensor(features_np, dtype=torch.float)
        mask_tensor = torch.tensor(mask_np, dtype=torch.bool)
        logits = trained_agent.forward_actor(features_tensor)
        logits_masked = masked_logits(logits, mask_tensor)
        dist = D.Categorical(logits=logits_masked)
        action = dist.sample()
        selected_actions.append(action.item())
        obs, reward, done, _ = env.step(action.item())

    print("选取的卫星索引：", selected_actions)
    print("最终奖励（实际 DGDOP）：", reward)
