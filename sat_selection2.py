#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from datetime import datetime, timezone, timedelta
import os

# 设置线程环境变量（Selection1 部分建议）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pylab import mpl

import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from skyfield.api import load, wgs84, EarthSatellite
from skyfield.framelib import itrs, tirs
import matplotlib.pyplot as plt

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
mpl.rcParams["axes.unicode_minus"] = False


# ------------------------------
# Selection1 部分：自定义 Satellite 及相关函数
# ------------------------------
class Satellite:
    def __init__(self, sat_object):
        """
        sat_object: Skyfield 中解析 TLE 后返回的 EarthSatellite 对象
        """
        self.sat_object = sat_object
        self.name = sat_object.name
        # 卫星的 ECEF 坐标（单位 m）
        self.position = None
        # 仰角、方位角（单位：度）
        self.elevation = None
        self.azimuth = None
        # GEV 特征参数：phi（垂直角，单位：弧度）、beta（水平特征角，单位：弧度）
        self.phi = None
        self.beta = None
        # 权重（用于排序备用）
        self.weight = None

    def __repr__(self):
        return f"{self.name}"


def update_satellite_info(sat, observer, t):
    """
    根据观测者位置 observer 更新卫星信息：
      - 获取卫星的 ECEF 坐标（单位：m）
      - 计算仰角、方位角（单位：度）
      - 根据仰角设置 phi：仰角>=50°时赋值 2π/3，否则赋值 π/3；beta 为方位角的弧度
    """
    pos = sat.sat_object.at(t).position
    sat.position = pos.m  # 单位 m
    difference = sat.sat_object - observer
    alt, az, _ = difference.at(t).altaz()
    sat.elevation = alt.degrees
    sat.azimuth = az.degrees
    if sat.elevation >= 50:
        sat.phi = 2 * np.pi / 3
    else:
        sat.phi = np.pi / 3
    sat.beta = np.radians(sat.azimuth)


def compute_dgdop(combo, user_pos):
    """
    计算给定卫星组合 combo 的 DGDOP 指标（单位 m）
    combo: 卫星列表（Satellite 对象）
    user_pos: 观测者位置（单位 m）
    """
    G = []
    for sat in combo:
        diff = sat.position - user_pos
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
        dop = np.sqrt(np.trace(Q))
    except np.linalg.LinAlgError:
        dop = np.inf
    return 10 * dop


def ossa_selection(satellites, user_pos, n):
    """
    OSSA 算法：遍历所有可能的 n 颗卫星组合，返回 DGDOP 最小的组合作为最佳方案
    """
    best_combo = None
    best_dop = np.inf
    for combo in combinations(satellites, n):
        dop_val = compute_dgdop(combo, user_pos)
        if dop_val < best_dop:
            best_dop = dop_val
            best_combo = combo
    return best_combo, best_dop


def maxele_selection(satellites, user_pos, n):
    """
    MaxEle 算法：将 360° 天区均分为 n 个区间，在每个区间内选取仰角最大的卫星
    """
    bins = np.linspace(0, 360, n + 1)
    selected = []
    for i in range(n):
        sats_in_bin = [sat for sat in satellites if bins[i] <= sat.azimuth < bins[i + 1]]
        if sats_in_bin:
            best_sat = max(sats_in_bin, key=lambda sat: sat.elevation)
            selected.append(best_sat)
    if len(selected) < n:
        remaining = [sat for sat in satellites if sat not in selected]
        remaining = sorted(remaining, key=lambda s: s.elevation, reverse=True)
        for sat in remaining:
            selected.append(sat)
            if len(selected) == n:
                break
    dop = compute_dgdop(selected, user_pos)
    return selected, dop


def fcsdp_selection(satellites, user_pos, n):
    """
    FCSDp 算法：基于分层聚类与加权匹配实现卫星选择。
    将卫星根据 phi 分为两组（上层与下层），各自利用 K-means 聚类后选出候选卫星，
    最后合并候选结果并调整至目标数量 n。
    """
    upper = [sat for sat in satellites if sat.phi > np.pi / 2]
    lower = [sat for sat in satellites if sat.phi <= np.pi / 2]

    upper_label1 = 2 * np.pi / 3
    lower_label1 = np.pi / 3

    if n % 2 == 0:
        k_upper = n // 2
        k_lower = n // 2
    else:
        k_upper = (n + 1) // 2
        k_lower = (n - 1) // 2

    selected_upper = []
    if upper and k_upper > 0:
        X_upper = np.array([[sat.beta] for sat in upper])
        kmeans_upper = KMeans(n_clusters=k_upper, random_state=0).fit(X_upper)
        centers_upper = kmeans_upper.cluster_centers_
        for clus_label in range(k_upper):
            indices = np.where(kmeans_upper.labels_ == clus_label)[0]
            best_sat = None
            best_weight = np.inf
            for i in indices:
                sat = upper[i]
                cur_weight = abs(sat.beta - centers_upper[clus_label][0]) + abs(sat.phi - upper_label1)
                if cur_weight < best_weight:
                    best_weight = cur_weight
                    best_sat = sat
            if best_sat is not None:
                selected_upper.append(best_sat)

    selected_lower = []
    if lower and k_lower > 0:
        X_lower = np.array([[sat.beta] for sat in lower])
        kmeans_lower = KMeans(n_clusters=k_lower, random_state=0).fit(X_lower)
        centers_lower = kmeans_lower.cluster_centers_
        for clus_label in range(k_lower):
            indices = np.where(kmeans_lower.labels_ == clus_label)[0]
            best_sat = None
            best_weight = np.inf
            for i in indices:
                sat = lower[i]
                cur_weight = abs(sat.beta - centers_lower[clus_label][0]) + abs(sat.phi - lower_label1)
                if cur_weight < best_weight:
                    best_weight = cur_weight
                    best_sat = sat
            if best_sat is not None:
                selected_lower.append(best_sat)

    selected = selected_upper + selected_lower
    if len(selected) < n:
        remaining = [sat for sat in satellites if sat not in selected]
        for sat in remaining:
            if sat.phi > np.pi / 2:
                sat.weight = abs(sat.phi - upper_label1)
            else:
                sat.weight = abs(sat.phi - lower_label1)
        remaining = sorted(remaining, key=lambda s: s.weight)
        while len(selected) < n and remaining:
            selected.append(remaining.pop(0))
    elif len(selected) > n:
        for sat in selected:
            if sat.phi > np.pi / 2:
                sat.weight = abs(sat.phi - upper_label1)
            else:
                sat.weight = abs(sat.phi - lower_label1)
        selected = sorted(selected, key=lambda s: s.weight)[:n]

    dop = compute_dgdop(selected, user_pos)
    return selected, dop


def visable(sat, observer, t):
    """
    判断 Satellite 对象是否可见（通过计算仰角 > 25°）
    """
    difference = sat.sat_object - observer
    topocentric = difference.at(t)
    alt, az, _ = topocentric.altaz()
    return alt.degrees > 25


# ------------------------------
# RL_GNN 测试部分：仅用于加载训练好的模型并进行推理
# ------------------------------
# 为了测试 RL_GNN，我们需要重新定义部分模型相关的函数与类

import torch
import torch.nn as nn
import torch.optim as optim


def compute_dgdop_combo(selected_positions, user_pos):
    """
    计算 DGDOP 值（数值越小表示定位几何越优）
    selected_positions: list，每个元素为一个卫星的 ECEF 坐标 (np.array, shape (3,))，单位 m
    user_pos: np.array，观测者位置（单位：m）
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
        dop = np.sqrt(np.trace(Q))
    except np.linalg.LinAlgError:
        dop = np.inf
    return 10 * dop


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        support = self.linear(x)
        out = torch.matmul(adj, support)
        return out


class GNNBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GNNBlock, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden_features)
        self.gc2 = GraphConvolution(hidden_features, out_features)

    def forward(self, x, adj):
        x = torch.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x


class PointerNet_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_select, num_layers=1, gnn_hidden_dim=None, gnn_out_dim=None):
        super(PointerNet_GNN, self).__init__()
        self.n_select = n_select
        if gnn_hidden_dim is None:
            gnn_hidden_dim = hidden_dim
        if gnn_out_dim is None:
            gnn_out_dim = input_dim
        self.gnn = GNNBlock(input_dim, gnn_hidden_dim, gnn_out_dim)
        self.encoder = nn.LSTM(gnn_out_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, bidirectional=True)
        self.init_h = nn.Linear(2 * hidden_dim, hidden_dim)
        self.init_c = nn.Linear(2 * hidden_dim, hidden_dim)
        self.decoder = nn.LSTMCell(2 * hidden_dim, hidden_dim)
        self.pointer = nn.Linear(hidden_dim, 2 * hidden_dim)

    def forward(self, inputs, adj):
        batch_size, m, _ = inputs.size()
        gnn_out = []
        for i in range(batch_size):
            xi = inputs[i]
            adji = adj[i]
            gnn_feature = self.gnn(xi, adji)
            gnn_out.append(gnn_feature)
        gnn_out = torch.stack(gnn_out, dim=0)
        encoder_outputs, _ = self.encoder(gnn_out)
        encoder_mean = encoder_outputs.mean(dim=1)
        h = self.init_h(encoder_mean)
        c = self.init_c(encoder_mean)
        decoder_input = encoder_mean
        selected_indices = []
        log_probs = []
        mask = torch.zeros(batch_size, m, device=inputs.device, dtype=torch.bool)
        for _ in range(self.n_select):
            h, c = self.decoder(decoder_input, (h, c))
            query = self.pointer(h)
            scores = torch.bmm(encoder_outputs, query.unsqueeze(2)).squeeze(2)
            scores = scores.masked_fill(mask, -1e9)
            probs = torch.softmax(scores, dim=1)
            dist = torch.distributions.Categorical(probs)
            selected = dist.sample()
            log_prob = dist.log_prob(selected)
            selected_indices.append(selected)
            log_probs.append(log_prob)
            mask = mask.clone()
            mask[torch.arange(batch_size), selected] = True
            decoder_input = encoder_outputs[torch.arange(batch_size), selected]
        selected_indices = torch.stack(selected_indices, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        total_log_probs = log_probs.sum(dim=1)
        return selected_indices, total_log_probs


def compute_adjacency(candidate_positions, threshold=1000.0):
    """
    candidate_positions: numpy 数组，形状 (m, 3)，单位 m
    threshold: 距离阈值（m）
    返回归一化的邻接矩阵，其中两颗卫星间若距离小于阈值则视为相邻
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
    deg = np.sum(adj, axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    adj = adj / deg
    return adj


def visable_rl(sat, observer, t):
    """
    判断 EarthSatellite 对象是否可见（仰角 > 25°）
    """
    difference = sat - observer
    topocentric = difference.at(t)
    alt, az, _ = topocentric.altaz()
    return alt.degrees > 25


# ------------------------------
# 主程序
# ------------------------------
if __name__ == '__main__':
    # ------------------------------
    # 基础设置与 TLE 数据加载
    # ------------------------------
    ts = load.timescale()
    t = ts.from_datetime(datetime(2025, 1, 6, 14, 0, 0, tzinfo=timezone.utc))
    tle_file = 'satellite_data/starlink_tle_20250106_144506.txt'
    satellites_tle = load.tle_file(tle_file)
    print(f"加载了 {len(satellites_tle)} 颗卫星 TLE 数据.")

    # 定义观测者
    observer = wgs84.latlon(45.75, 126.68, elevation_m=100)
    user_pos = observer.at(t).position.m  # 单位 m

    # ------------------------------
    # Selection1 部分：转换为 Satellite 对象并更新信息
    # ------------------------------
    satellites_sel = [Satellite(sat) for sat in satellites_tle]
    satellites_sel = [sat for sat in satellites_sel if visable(sat, observer, t)]
    print(f"[Selection1] 可见卫星数量: {len(satellites_sel)}")
    for sat in satellites_sel:
        update_satellite_info(sat, observer, t)

    n = 4  # 选择卫星数量

    # OSSA 算法
    start = time.perf_counter()
    ossa_combo, ossa_dop = ossa_selection(satellites_sel, user_pos, n)
    elapsed_ossa = (time.perf_counter() - start) * 1000
    ossa_names = [sat.name for sat in ossa_combo]
    print("\n[OSSA] 卫星组合:", ossa_names)
    print(f"[OSSA] DGDOP = {ossa_dop:.2f}, 时间 = {elapsed_ossa:.4f} ms")

    # MaxEle 算法
    start = time.perf_counter()
    maxele_combo, maxele_dop = maxele_selection(satellites_sel, user_pos, n)
    elapsed_maxele = (time.perf_counter() - start) * 1000
    maxele_names = [sat.name for sat in maxele_combo]
    print("\n[MaxEle] 卫星组合:", maxele_names)
    print(f"[MaxEle] DGDOP = {maxele_dop:.2f}, 时间 = {elapsed_maxele:.4f} ms")

    # FCSDp 算法
    start = time.perf_counter()
    fcsdp_combo, fcsdp_dop = fcsdp_selection(satellites_sel, user_pos, n)
    elapsed_fcsdp = (time.perf_counter() - start) * 1000
    fcsdp_names = [sat.name for sat in fcsdp_combo]
    print("\n[FCSDp] 卫星组合:", fcsdp_names)
    print(f"[FCSDp] DGDOP = {fcsdp_dop:.2f}, 时间 = {elapsed_fcsdp:.4f} ms")

    # ------------------------------
    # RL_GNN 测试部分：加载训练好的模型并进行推理
    # ------------------------------
    # 从 TLE 数据中过滤出满足仰角>25°的 EarthSatellite 对象（用于 RL_GNN 测试）
    rl_satellites = [sat for sat in satellites_tle if visable_rl(sat, observer, t)]
    m_rl = len(rl_satellites)

    # 构造候选卫星的特征与位置（单位均为 m 或弧度）
    feature_dim = 2
    candidate_positions = np.empty((m_rl, 3))
    candidate_features = np.empty((m_rl, feature_dim))
    for i, sat in enumerate(rl_satellites):
        candidate_positions[i] = sat.at(t).position.m
        diff = sat - observer
        topocentric = diff.at(t)
        alt, az, _ = topocentric.altaz()
        elevation = alt.degrees
        phi = 2 * np.pi / 3 if elevation >= 50 else np.pi / 3
        beta = np.radians(az.degrees)
        candidate_features[i, 0] = phi
        candidate_features[i, 1] = beta

    candidate_features = torch.tensor(candidate_features, dtype=torch.float)

    # 加载训练好的 RL_GNN 模型
    n_select = 4
    hidden_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rl_model = PointerNet_GNN(input_dim=feature_dim, hidden_dim=hidden_dim, n_select=n_select)
    rl_model.to(device)
    model_save_path = 'pointer_net_gnn.pth'
    rl_model.load_state_dict(torch.load(model_save_path,weights_only=True, map_location=device))
    rl_model.eval()

    # 前向推理前热身并同步 GPU
    with torch.no_grad():
        inputs = candidate_features.unsqueeze(0).to(device)
        adj = compute_adjacency(candidate_positions, threshold=1000.0)
        adj = torch.tensor(adj, dtype=torch.float, device=device).unsqueeze(0)
        _ = rl_model(inputs, adj)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    with torch.no_grad():
        inputs = candidate_features.unsqueeze(0).to(device)
        adj = compute_adjacency(candidate_positions, threshold=1000.0)
        adj = torch.tensor(adj, dtype=torch.float, device=device).unsqueeze(0)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        selected_indices, _ = rl_model(inputs, adj)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed_rl = (time.perf_counter() - start) * 1000  # ms
        selected_indices = selected_indices.cpu().numpy().tolist()[0]
        rl_names = [rl_satellites[j].name for j in selected_indices]
        sel_positions_rl = [candidate_positions[j] for j in selected_indices]
        rl_dop = compute_dgdop_combo(sel_positions_rl, user_pos)

    print("\n[RL-GNN] 卫星组合:", rl_names)
    print(f"[RL-GNN] DGDOP = {rl_dop:.2f}, 时间 = {elapsed_rl:.4f} ms")

    # ------------------------------
    # 综合对比结果与绘图
    # ------------------------------
    method_names = ['RL-GNN', 'OSSA', 'FCSDp', 'MaxEle']
    dop_values = [rl_dop, ossa_dop, fcsdp_dop, maxele_dop]
    time_values = [elapsed_rl, elapsed_ossa, elapsed_fcsdp, elapsed_maxele]

    print("\n===== 综合对比结果 =====")
    for i, method in enumerate(method_names):
        if method == 'RL-GNN':
            names = rl_names
        elif method == 'OSSA':
            names = ossa_names
        elif method == 'FCSDp':
            names = fcsdp_names
        else:
            names = maxele_names
        print(f"{method}: 组合 = {names}, DGDOP = {dop_values[i]:.2f}, 时间 = {time_values[i]:.4f} ms")

    x = np.arange(len(method_names))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # DGDOP 对比
    axes[0].bar(x, dop_values, width, color=['skyblue', 'salmon', 'lightgreen', 'plum'],label=method_names)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(method_names)
    axes[0].set_ylabel('DGDOP')
    axes[0].set_title('DGDOP')
    axes[0].legend()

    # 执行时间对比
    axes[1].set_yscale('log')
    axes[1].bar(x, time_values, width, color=['skyblue', 'salmon', 'lightgreen', 'plum'],label=method_names)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(method_names)
    axes[1].set_ylabel('time (ms)')
    axes[1].set_title('time')
    axes[1].legend()

    plt.tight_layout()
    plt.show()