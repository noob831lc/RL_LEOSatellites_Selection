import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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