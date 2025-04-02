import torch

# 创建一个行向量
x = torch.arange(12, dtype=torch.float32)
print(x) 
print(x.shape)      # 张量（沿每个轴的长度）的形状
print(x.numel())    # 张量中元素的总数（张量的大小），形状的所有元素乘积

# 改变张量的形状，但不影响张量的大小（张量元素个数）
X = x.reshape(3, 4)
print(X)
print(X.shape)
print(X.numel())

# 自动计算维度
X1 = x.reshape(-1, 4)
X2 = x.reshape(3, -1)
print(X1.shape)
print(X2.shape)

# 全0
X = torch.zeros((2, 3, 4))
print(X)

# 全1
X = torch.ones((2, 3, 4))
print(X)

# 标准高斯分布（正态分布）
X = torch.randn(3, 4)
print(X)

# Python列表转Tensor
X = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(X)

# 按元素计算
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)