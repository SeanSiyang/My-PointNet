# 4.2.1 均方误差
import os
import sys
from dataset.mnist import load_mnist
import numpy as np
print("-------------- 4.2.1 ----------------")


def mean_squared_error(output, label):
    return 0.5 * np.sum((output - label) ** 2)


label = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
output = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(np.array(output), np.array(label)))

output = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(mean_squared_error(np.array(output), np.array(label)))

# 4.2.2 交叉熵误差
print("-------------- 4.2.2 ----------------")


def cross_entropy_error(output, label):
    delta = 1e-7
    return -np.sum(label * np.log(output + delta))


label = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
output = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(output), np.array(label)))

output = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(output), np.array(label)))


# 4.2.3 mini-batch学习
print("-------------- 4.2.3 ----------------")
sys.path.append(os.pardir)

(imgs_train, labels_train), (imgs_test, labels_test) = load_mnist(
    normalize=True,
    one_hot_label=True
)

print(imgs_train.shape)
print(labels_train.shape)

# 4.2.4 mini-batch学习
print("-------------- 4.2.4 ----------------")


def cross_entropy_error_batch(output, label):
    # 单个数据的交叉熵损失
    if output.dim == 1:
        output = output.reshape(1, output.size)
        label = label.reshape(1, label.size)

    batch_size = output.shape[0]

    return -np.sum(label * np.log(output + 1e-7)) / batch_size


def cross_entropy_error_batch_not_hot(output, label):
    if output.dim == 1:
        output = output.reshape(1, output.size)
        label = label.reshape(1, label.size)

    batch_size = output.shape[0]

    # 将正确标签索引对应位置的输出提取出来计算即可，这是由于交叉熵损失的计算过程决定的
    return -np.sum(np.log(output[np.arange(0, batch_size), label] + 1e-7)) / batch_size


# 4.3.1 导数（数值微分：利用微小的差分求导）
print("-------------- 4.3.1 ----------------")
def numerical_diff(function, input):
    h = 1e-4
    return (function(input + h) - function(input - h)) / (2 * h)

# 4.3.2 数值微分的例子
print("-------------- 4.3.2 ----------------")
def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
# plt.show()

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))

def tangent_line(function, x):
    d = numerical_diff(function, x)
    print(d)
    y = function(x) - d*x   # 从切线的标准形式可以推导出斜截式，进而得到截距
    return lambda t: d*t + y

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y2)
# plt.show()

# 4.3.3 偏导数
print("-------------- 4.3.3 ----------------")
from mpl_toolkits.mplot3d import Axes3D

x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)

# 创建网格坐标矩阵
X, Y = np.meshgrid(x0, x1)

# 计算函数值
Z = X**2 + Y**2

# 创建画布
fig = plt.figure(figsize=(12, 8))

# 创建3D曲面图
ax1 = fig.add_subplot(121, projection='3d')  # 1行2列的第1个位置
# cmap是颜色映射，alpha是透明度，plot_surface是创建三维曲面
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8) 

# 设置坐标轴标签
ax1.set_xlabel('x0')
ax1.set_ylabel('x1')
ax1.set_zlabel('f(x0, x1)')
ax1.set_title('3D Surface Plot of $f(x_0, x_1) = x_0^2 + x_1^2$')

# 添加颜色条
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

# 创建等高线图（二维表示）
ax2 = fig.add_subplot(122)  # 1行2列的第2个位置
contour = ax2.contourf(X, Y, Z, 20, cmap='viridis')  # 20个颜色层次

# 添加等高线标签
ax2.contour(X, Y, Z, 10, colors='black', linewidths=0.5)  # 10条等高线

# 设置坐标轴标签和标题
ax2.set_xlabel('x0')
ax2.set_ylabel('x1')
ax2.set_title('Contour Plot of $f(x_0, x_1) = x_0^2 + x_1^2$')

# 添加颜色条
fig.colorbar(contour, ax=ax2, shrink=0.5, aspect=5)

# 调整布局并显示图形
plt.tight_layout()
# plt.show()

def function_2(x):
    # return np.sum(x**2)
    return x[0] ** 2 + x[1] ** 2

# x0 = 3, x1 = 4
def function_tmp1(x0):
    return x0 ** 2.0 + 4.0 ** 2.0

def function_tmp2(x1):
    return 3.0 ** 2.0 + x1 ** 2.0

print(numerical_diff(function_tmp1, 3.0))
print(numerical_diff(function_tmp2, 4.0))

# 4.4 梯度
print("-------------- 4.4 ----------------")

def numerical_gradient(function, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = function(x)
        
        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = function(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
    
    return grad

print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))