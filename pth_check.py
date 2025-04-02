import torch  # 命令行是逐行立即执行的
from RL_GNN import PointerNet_GNN

# content = torch.load('pointer_net_gnn.pth', weights_only=True)
# n_select = 4
# hidden_dim = 128
# feature_dim = 2
# model = PointerNet_GNN(input_dim=feature_dim, hidden_dim=hidden_dim, n_select=n_select)
# params = model.load_state_dict(content)
#
# print(params)
checkpoint = torch.load('pointer_net_gnn.pth', weights_only=True, map_location=torch.device('cpu'))

# 如果是模型状态字典，通常是个字典类型，可以遍历查看各层的参数
# print(checkpoint)

# 若仅想查看某些特定信息，例如:
if isinstance(checkpoint, dict):
    for k in checkpoint.keys():
        print(k)
