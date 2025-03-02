import numpy as np
import os
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset

# 设置常数
NUM_LABELS = 10
NUM_CLIENTS = 10

# 数据集存储路径
data_path = os.path.join(os.getcwd(), "cifar10")
if not os.path.exists(data_path):
    os.makedirs(data_path)

# 下载 CIFAR-10 数据集
train_data = CIFAR10(data_path, train=True, download=True)
data, target = np.array(train_data.data), np.array(train_data.targets)

# 创建用于保存每个客户数据的路径
save_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), f"iid_{NUM_CLIENTS}")
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 数据集的总样本数量
num_examples = len(train_data)
num_examples_per_client = num_examples // NUM_CLIENTS
print(f"Total examples: {num_examples}, examples per client: {num_examples_per_client}")

# 打乱数据顺序
perm = np.random.permutation(num_examples)

# 将数据划分为多个客户端
for cid in range(NUM_CLIENTS):
    indices = perm[cid * num_examples_per_client : (cid + 1) * num_examples_per_client]
    
    # 获取客户端数据
    client_X = data[indices]
    client_y = target[indices]
    
    # 合并数据
    combined = list(zip(client_X, client_y))

    # 保存每个客户端的数据
    cname = f'client{cid}'
    np.save(os.path.join(save_path, f"{cname}.npy"), combined)

print("Federated datasets created successfully!")
