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
    # 求单个数据的交叉熵损失，需要统一数据的形状
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
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)
        # return x[0] ** 2 + x[1] ** 2

# x0 = 3, x1 = 4
def function_tmp1(x0):
    return x0 ** 2.0 + 4.0 ** 2.0

def function_tmp2(x1):
    return 3.0 ** 2.0 + x1 ** 2.0

print(numerical_diff(function_tmp1, 3.0))
print(numerical_diff(function_tmp2, 4.0))

# 4.4 梯度
print("-------------- 4.4 ----------------")

def _numerical_gradient_no_batch(function, x):
    """
    这里处理的权重是向量，如果是矩阵，则无法处理
    此处的batch并不是batch_size
    """
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)
        x[idx] = float(tmp_val) + h
        fxh1 = function(x)
        
        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = function(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
    
    return grad

def numerical_gradient(function, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(function, X)
    else:
        grad = np.zeros_like(X)
        # 遍历权重矩阵，逐行计算梯度
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(function, x)
        
        return grad

print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))

x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)
X, Y = np.meshgrid(x0, x1)

X = X.flatten()
Y = Y.flatten()

grad = numerical_gradient(function_2, np.array([X, Y]))

plt.figure()        # 创建新的画布
# 绘制二维向量场
plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")

plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid()          # 显示网格
plt.legend()        # 显示图例
plt.draw()          # 重新绘制当前图形
# plt.show()          # 阻塞程序执行，显示所有图形

# 4.4.1 梯度法
print("-------------- 4.4.1 ----------------")
def gradient_descent(function, init_weight, lr=0.01, step_nums=100):
    weight = init_weight
    weight_history = []
    
    for i in range(step_nums):
        weight_history.append(weight.copy())
        
        grad = numerical_gradient(function, weight)
        weight -= lr * grad
    
    return weight, np.array(weight_history)

init_weight = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_weight=init_weight, lr=0.1, step_nums=100))

weight, weight_history = gradient_descent(
    function=function_2,
    init_weight=init_weight,
    lr=0.1,
    step_nums=100
)

plt.figure()                        # 创建新的画布
# x坐标范围-5到5，y坐标轴始终为0，--b蓝色虚线样式
plt.plot([-5, 5], [0, 0], '--b')    # 绘制x轴参考线
plt.plot([0, 0], [-5, 5], '--b')    # 绘制y轴参考线
plt.plot(weight_history[:, 0], weight_history[:, 1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")

# plt.show()

grad_1, _ = gradient_descent(function_2, init_weight=init_weight, lr=10.0, step_nums=100)
grad_2, _ = gradient_descent(function_2, init_weight=init_weight, lr=1e-10, step_nums=100)
print(grad_1)
print(grad_2)

# 4.4.2 神经网络的梯度
print("-------------- 4.4.2 ----------------")
from common.functions import softmax, cross_entropy_error_class_label
from common.gradient import numerical_gradient_official

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, label):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, label)

        return loss

net = SimpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
# 最大值索引
print(np.argmax(p))

label = np.array([0, 0, 1])
print(net.loss(x, label))

def f(W):
    """
    此处为什么有W，因为为了传入numerical_gradient的时候
    
    """
    return net.loss(x, label)

dW = numerical_gradient_official(f, net.W)
print(dW)

# 4.5.1 2层神经网络的类
print("-------------- 4.5.1 ----------------")
from common.functions import sigmoid, sigmoid_grad

class TwoLayerNet:
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        output_size,
        weight_init_std=0.01
    ):
        """
        初始化
        input_size: 输入层的神经元数
        hidden_size: 隐藏层的神经元数
        """
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def predict(self, x):
        """
        推理
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
    
    def loss(self, input, label):
        """
        计算损失函数的值
        """
        y = self.predict(input)
        
        return cross_entropy_error(y, label)
    
    def accuracy(self, input, label):
        """
        计算识别精度
        """
        y = self.predict(input)
        y = np.argmax(y, axis=1)
        label = np.argmax(label, axis=1)
        
        accuracy = np.sum(y == label) / float(input.shape[0])
        
        return accuracy
    
    def numerical_gradient(self, input, label):
        """
        计算权重参数的梯度
        """
        loss_W = lambda W: self.loss(input, label)
        
        grads = {}
        grads['W1'] = numerical_gradient_official(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient_official(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient_official(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient_official(loss_W, self.params['b2'])
        
        return grads
    
    def gradient(self, input, label):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        grads = {}
        
        batch_num = input.shape[0]
        
        # forward
        a1 = np.dot(input, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - label) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(input.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

print(net.params['W1'].shape)
print(net.params['b1'].shape)
print(net.params['W2'].shape)
print(net.params['b2'].shape)


input = np.random.rand(100, 784)
label = np.random.rand(100, 10)

grads = net.numerical_gradient(input, label)

print(grads['W1'].shape)
print(grads['b1'].shape)
print(grads['W2'].shape)
print(grads['b2'].shape)

# 4.5.2 mini-batch的实现
print("-------------- 4.5.2 ----------------")

from dataset.mnist import load_mnist

(input_train, label_train), (input_test, label_test) = \
    load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []


# 超参数
iters_num = 10000
train_size = input_train.shape[0]
batch_size = 100
learning_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iter_per_batch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    input_batch = input_train[batch_mask]
    label_batch = label_train[batch_mask]
    
    # 计算梯度
    # grad = network.numerical_gradient(input_batch, label_batch)
    grad = network.gradient(input_batch, label_batch)
    
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 记录学习过程
    loss = network.loss(input_batch, label_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_batch == 0:
        train_acc = network.accuracy(input_train, label_train)
        test_acc = network.accuracy(input_test, label_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
    
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()