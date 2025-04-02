import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

# 这里假设我们能从 FCSDp 实现的模块中导入 compute_dgdop 函数
# 该函数用于根据所选卫星列表和观测者位置计算 DGDOP 值
# 如果没有模块化，可以将 selection1.py 中的 compute_dgdop 函数复制到此处
from selection1 import compute_dgdop


#############################################
# 1. 模拟卫星与环境的定义
#############################################
class SimulatedSatellite:
    """
    用于模拟的卫星数据结构，仅需要具备 compute_dgdop 函数所用的属性：
      - name
      - position：3D 坐标（km）
      - phi：GEV 垂直角（弧度）
      - beta：水平角（弧度）
      - elevation：仰角（度）
    """

    def __init__(self, sat_id, position, phi, beta, elevation):
        self.name = f"Sat-{sat_id}"
        self.position = position  # numpy 数组，shape: (3,)
        self.phi = phi
        self.beta = beta
        self.elevation = elevation

    def __repr__(self):
        return self.name


class SatelliteSelectionEnvRL:
    """
    简化的卫星选星环境：
      - 初始化时随机生成一定数量的模拟卫星（特征：[phi, beta, elevation]）
      - 状态为所有卫星的特征矩阵，以及一个 bool 型 mask（True 表示该卫星可选）
      - 每一步选择一颗卫星，当累计选择数量达到 n_select 后，
        根据所选卫星组合计算 DGDOP，取负值作为最终 reward。
    """

    def __init__(self, num_satellites=30, n_select=4, user_pos=np.array([0.0, 0.0, 0.0])):
        self.num_satellites = num_satellites
        self.n_select = n_select
        self.user_pos = user_pos  # 观测者位置（ITRS 坐标，单位 km）；这里简化为原点
        # 生成模拟卫星列表
        self.satellites = self._generate_satellites(num_satellites)
        self.feature_dim = 3  # 使用 [phi, beta, elevation] 作为输入特征
        self.features = self._get_features()
        # mask 用于表示尚未被选取的卫星
        self.mask = np.ones(self.num_satellites, dtype=bool)
        self.selected_idx = []

    def _generate_satellites(self, num):
        sats = []
        for i in range(num):
            # 模拟卫星位置（单位：km）
            position = np.random.uniform(20000, 40000, size=3)
            # 随机生成仰角，模拟范围 20° 到 90°
            elevation = np.random.uniform(20, 90)
            # 根据仰角设置 GEV 垂直角 phi：
            # 仰角 >= 50° 取 2π/3，否则取 π/3
            phi = 2 * np.pi / 3 if elevation >= 50 else np.pi / 3
            # 水平角 beta 均匀分布在 [0, 2π]
            beta = np.random.uniform(0, 2 * np.pi)
            sat = SimulatedSatellite(i, position, phi, beta, elevation)
            sats.append(sat)
        return sats

    def _get_features(self):
        """
        返回所有卫星的特征矩阵，shape: (num_satellites, feature_dim)
        特征顺序：[phi, beta, elevation]
        """
        feats = []
        for sat in self.satellites:
            feats.append([sat.phi, sat.beta, sat.elevation])
        return np.array(feats, dtype=np.float32)

    def reset(self):
        """
        重置环境，恢复所有卫星可选状态
        """
        self.mask = np.ones(self.num_satellites, dtype=bool)
        self.selected_idx = []
        # 若有需要，可在每个 episode 中重新生成卫星（此处保持固定）
        return self.features, self.mask

    def step(self, action):
        """
        执行动作：选取某颗卫星（动作为卫星索引）
        当累计选择数量达到 n_select 时，计算卫星组合的 DGDOP，
        并取负值作为最终 reward。
        """
        if not self.mask[action]:
            raise ValueError("所选卫星已被选取！")
        self.selected_idx.append(action)
        self.mask[action] = False
        done = len(self.selected_idx) >= self.n_select
        reward = 0.0
        if done:
            selected_sats = [self.satellites[i] for i in self.selected_idx]
            dop = compute_dgdop(selected_sats, self.user_pos)
            reward = -dop  # 奖励为负的 DGDOP，DGDOP 越低奖励越高
        return (self.features, self.mask), reward, done, {}


#############################################
# 2. PPO 网络及辅助函数
#############################################
def masked_logits(logits, mask_tensor):
    """
    对 logits 应用 mask，将已选择（mask=False）的卫星位置置为一个极小值，
    确保 softmax 时概率为 0。
    """
    neg_inf = -1e8
    return logits.masked_fill(~mask_tensor, neg_inf)


class PPONetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        """
        构造具有 Actor 和 Critic 网络结构的 PPO 模型
        """
        super(PPONetwork, self).__init__()
        # Actor 网络：对每颗卫星的特征输出一个标量得分
        self.actor_fc1 = nn.Linear(feature_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, 1)
        # Critic 网络：基于当前全局状态（这里采用所有尚未被选卫星特征的平均值）输出状态价值
        self.critic_fc1 = nn.Linear(feature_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, 1)

    def forward_actor(self, features):
        # features: tensor，shape: (N, feature_dim)
        x = torch.relu(self.actor_fc1(features))
        logits = self.actor_fc2(x).squeeze(-1)  # 输出 shape: (N,)
        return logits

    def forward_critic(self, features, mask):
        """
        根据当前状态（所有卫星特征及其 mask）输出一个全局状态价值。
        这里简单采用所有可选卫星（mask=True）特征的加权平均作为全局状态描述。
        """
        mask = mask.unsqueeze(1)  # 形状: (N, 1)
        mask_float = mask.float()
        if mask_float.sum() > 0:
            x = (features * mask_float).sum(dim=0, keepdim=True) / mask_float.sum()
        else:
            x = features.mean(dim=0, keepdim=True)
        x = torch.relu(self.critic_fc1(x))
        value = self.critic_fc2(x)
        return value.squeeze(0)


#############################################
# 3. PPO 算法的训练实现
#############################################
def ppo_update(network, optimizer, batch_states, batch_masks, batch_actions, batch_logprobs, batch_returns,
               batch_advantages,
               clip_epsilon=0.2, critic_coef=0.5, entropy_coef=0.01, epochs=4, mini_batch_size=32):
    """
    对收集到的 batch 数据进行 PPO 更新
    """
    batch_size = len(batch_states)
    indices = np.arange(batch_size)
    for epoch in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mb_idx = indices[start:end]
            # 对 mini-batch 数据转换为 tensor 格式
            states_mb = [batch_states[i] for i in mb_idx]  # 每项为 (features, mask)
            features_mb = torch.stack([torch.tensor(s[0]) for s in states_mb])  # shape: (batch, N, feature_dim)
            masks_mb = torch.stack([torch.tensor(s[1], dtype=torch.bool) for s in states_mb])  # shape: (batch, N)
            actions_mb = torch.tensor([batch_actions[i] for i in mb_idx])
            old_logprobs_mb = torch.tensor([batch_logprobs[i] for i in mb_idx])
            returns_mb = torch.tensor([batch_returns[i] for i in mb_idx],dtype=torch.float)
            advantages_mb = torch.tensor([batch_advantages[i] for i in mb_idx],dtype=torch.float)
            new_logprobs = []
            values = []
            entropies = []
            # 针对 mini-batch 中的每个样本分别计算新策略的 log_prob 与状态价值
            for i in range(features_mb.shape[0]):
                feats = features_mb[i]  # shape: (N, feature_dim)
                mask = masks_mb[i]  # shape: (N,)
                logits = network.forward_actor(feats)  # shape: (N,)
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
            # 计算策略更新比率
            ratio = torch.exp(new_logprobs - old_logprobs_mb)
            surr1 = ratio * advantages_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            # value_loss = torch.nn.functional.mse_loss(values, returns_mb)
            value_loss = torch.nn.functional.mse_loss(values.squeeze(-1), returns_mb)
            entropy_loss = -entropies.mean()
            loss = policy_loss + critic_coef * value_loss + entropy_coef * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_ppo(agent, env, num_epochs=1000, batch_size=64, clip_epsilon=0.2):
    """
    PPO 主训练函数：
      - 在每个 episode 中，记录每一步的状态/动作/奖励等数据
      - 每收集 batch_size 个 episode 后，进行一次 PPO 更新
    """
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    gamma = 0.99
    all_episode_rewards = []
    # 用于存放 batch 数据
    batch_states = []
    batch_actions = []
    batch_logprobs = []
    batch_returns = []
    batch_advantages = []
    episode_count = 0

    while episode_count < num_epochs:
        # 收集一个 episode（选星过程共 n_select 步）
        states = []
        actions = []
        logprobs = []
        rewards = []
        values = []

        obs = env.reset()  # obs 为 (features, mask)
        done = False
        episode_reward = 0
        while not done:
            features_np, mask_np = obs
            features_tensor = torch.tensor(features_np,dtype=torch.float)
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

        # 计算折扣累积回报（由于只有终端奖励，前面步骤 reward=0）
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

        # 每收集 batch_size 个 episode 后更新一次 PPO 模型
        if episode_count % batch_size == 0:
            ppo_update(agent, optimizer, batch_states,
                       [s[1] for s in batch_states],  # 提取每个 state 中的 mask
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
# 4. 主程序：训练和测试 PPO 选星算法
#############################################
if __name__ == "__main__":
    # 设置随机种子，保证实验可重复
    np.random.seed(42)
    torch.manual_seed(42)

    # 定义环境参数
    num_satellites = 25
    n_select = 4
    # 观测者位置（ITRS 坐标，这里简单取原点）
    user_pos = np.array([0.0, 0.0, 0.0])
    env = SatelliteSelectionEnvRL(num_satellites=num_satellites, n_select=n_select, user_pos=user_pos)

    # 定义智能体（输入特征维数为 3）
    feature_dim = 3
    agent = PPONetwork(feature_dim, hidden_dim=64)

    # 离线训练 PPO 模型（示例：训练 500 个 episode，每 16 个 episode 更新一次）
    trained_agent, rewards_history = train_ppo(agent, env, num_epochs=500, batch_size=16, clip_epsilon=0.2)

    # 测试训练后的智能体（在线选星）
    obs = env.reset()
    done = False
    selected_actions = []
    while not done:
        features_np, mask_np = obs
        features_tensor = torch.tensor(features_np,dtype=torch.float)
        mask_tensor = torch.tensor(mask_np, dtype=torch.bool)
        logits = trained_agent.forward_actor(features_tensor)
        logits_masked = masked_logits(logits, mask_tensor)
        dist = D.Categorical(logits=logits_masked)
        action = dist.sample()
        selected_actions.append(action.item())
        obs, reward, done, _ = env.step(action.item())

    print("选取的卫星索引：", selected_actions)
    print("最终奖励（DGDOP）：", -reward)