import torch
import numpy as np
import matplotlib.pyplot as plt  # 用于绘图
from typing import List, Any, Union
from torch import nn, optim
from torch import Tensor
from model import compute_adjacency
from algorithm.Selection import compute_dgdop


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int,
    batch_size: int,
    candidate_features: Tensor,
    candidate_positions: np.ndarray,
    user_pos: np.ndarray
) -> nn.Module:
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
    adj: np.ndarray = compute_adjacency(candidate_positions, threshold=1000.0)
    adj = torch.tensor(adj, dtype=torch.float, device=device)
    adj = adj.unsqueeze(0).repeat(batch_size, 1, 1)

    reward_history: List[float] = []
    loss_history: List[float] = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        # 扩展 candidate_features 至 batch 维度：(batch_size, m, feature_dim)
        inputs = candidate_features.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        selected_indices, log_probs = model(inputs, adj)

        # 计算每个样本的奖励（使用 DGDOP 作为指标：reward = -DGDOP）
        rewards: List[float] = []
        selected_indices_np = selected_indices.cpu().detach().numpy()
        for i in range(batch_size):
            indices = selected_indices_np[i].tolist()
            sel_positions = [candidate_positions[j] for j in indices]
            dop = compute_dgdop(sel_positions, user_pos)
            reward = 1 / (1 + dop)  # DGDOP 越低，reward 越高
            rewards.append(reward)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)

        loss = - (log_probs * rewards_tensor).mean()
        loss.backward()
        optimizer.step()

        mean_reward = rewards_tensor.mean().item()
        loss_val = loss.item()
        reward_history.append(mean_reward)
        loss_history.append(loss_val)

        print(f"Epoch {epoch}: Loss = {loss_val:.4f}, Mean Reward = {mean_reward:.4f}")

    # 绘制训练曲线
    window: int = 100  # 设置窗口大小，根据数据量调整最佳取值
    smooth_loss: np.ndarray = np.convolve(loss_history, np.ones(window) / window, mode='valid')
    smooth_reward: np.ndarray = np.convolve(reward_history, np.ones(window) / window, mode='valid')
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