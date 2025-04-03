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

# 张量连结concatenate
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

print(X.shape)
print(Y.shape)

X1 = torch.cat((X, Y), dim=0)
print(X1)
print(X1.shape)

X2 = torch.cat((X, Y), dim=1)
print(X2)
print(X2.shape)

X = torch.arange(12).reshape((3, 4))
print(X)
X[1, 2] = 9
print(X)
X[1][3] = 9
print(X)

Z = torch.zeros_like(Y)
print('id(Z): ', id(Z))
Z[:] = X + Y
print('id(Z): ', id(Z))

# tensor与Numpy之间的切换
A = X.numpy()
B = torch.tensor(A)
print(type(A))
print(type(B))

# 如果张量为一个标量，可以使用item方法或者直接类型转换
a = torch.tensor([3.5])
print(a)
print(a.item())
print(float(a))
print(int(a))

print(X == Y)
print(X < Y)
print(X > Y)

X = torch.arange(24, dtype=torch.float32).reshape((3, 2, 4))
Y = torch.tensor([[[2.0, 1, 4, 3]], [[1, 2, 3, 4]], [[4, 3, 2, 1]]])
print(X.shape)
print(Y.shape)
print(X + Y)
print((X + Y).shape)