# 3.2.2 阶跃函数的实现
print("-------------- 3.2.2 -------------------")
import numpy as np
def step_function_for_double(x):
    if x > 0:
        return 1
    else:
        return 0

# 接收Numpy数组的参数，调用方式是 step_function_for_numpy(np.array([1.0, 2.0]))
def step_function_for_numpy(x):
    # y = x[x > 0]
    y = x > 0
    # return y.astype(np.int)
    return y.astype(int)    # 转换Numpy数组中元素的类型

a = np.array([0.5, 1.0])
step_function_for_numpy(a)

# 3.2.3 阶跃函数的图形
print("-------------- 3.2.3 -------------------")
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, dtype=int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

# plt.plot(x, y)
# plt.ylim(-0.1, 1.1) # 指定y轴的范围
# plt.show()

# 3.2.4 sigmoid函数的实现
print("-------------- 3.2.4 -------------------")
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

# 3.2.7 ReLU函数
print("-------------- 3.2.7 -------------------")
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 1.0)
y = relu(x)

# plt.plot(x, y)
# plt.show()

# 3.3.1 多维数组
print("-------------- 3.3.1 -------------------")

# 一维数组
A = np.array([1, 2, 3, 4])
print(A)

print(np.ndim(A))
print(A.shape)
print(A.shape[0])

# 二维数组
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)

print(np.ndim(B))
print(B.shape)

# 3.3.2 矩阵乘法
print("-------------- 3.3.2 -------------------")
A = np.array([[1, 2], [3, 4]])
print(A.shape)
B = np.array([[5, 6], [7, 8]])
print(B.shape)

print(np.dot(A, B))

# 3.3.3 神经网络的内积
print("-------------- 3.3.3 -------------------")
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])

print(X.shape)
print(W.shape)

Y = np.dot(X, W)
print(Y)

# 3.4.2 各层间信号传递的实现
print("-------------- 3.4.2 -------------------")

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1

Z1 = sigmoid(A1)
print(A1)
print(Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B2
Y = identity_function(A3)

# 3.4.3 代码实现小结
print("-------------- 3.4.3 -------------------")
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)

# 3.5.1 恒等函数和softmax函数
print("-------------- 3.5.1 -------------------")

a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a

print(y)

def softmax_1(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

a = np.array([1010, 1000, 990])
print(np.exp(a) / np.sum(np.exp(a)))

c = np.max(a)
print(a - c)

print(np.exp(a - c) / np.sum(np.exp(a - c)))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

# 3.5.3 恒等函数和softmax函数
print("-------------- 3.5.3 -------------------")

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))

# 3.6.1 MNIST数据集
print("-------------- 3.6.1 -------------------")
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(train_img, train_label), (test_img, test_label) = load_mnist(
    flatten=True, normalize=False, one_hot_label=False)

print(train_img.shape)
print(train_label.shape)
print(test_img.shape)
print(test_label.shape)

img = train_img[0]
label = train_label[0]

print(label)
print(img.shape)
img = img.reshape(28, 28)
print(img.shape)
# img_show(img)

# 3.6.2 神经网络的推理处理
print("-------------- 3.6.2 -------------------")
import pickle

def get_data():
    (train_img, train_label), (test_img, test_label) = load_mnist(
    flatten=True, normalize=True, one_hot_label=False)
    
    return test_img, test_label

def init_network_predict(pkl_path):
    # 检查文件是否存在
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"weight file didn't find: {pkl_path}")
    
    # 检查文件是否为pickle格式
    if not pkl_path.lower().endswith('.pkl'):
        raise Warning(f"警告: 文件扩展名不是.pkl: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        try:
            network = pickle.load(f)
        except(pickle.UnpicklingError, EOFError, AttributeError, ImportError,
                IndexError) as e:
            raise pickle.UnpicklingError(f"Pickle解析失败: {str(e)}")
    
    # 验证加载内容的类型和结构
    if not isinstance(network, dict):
        raise TypeError(f"加载的内容不是字典类型: {type(network)}")
    
    # 检查关键权重是否存在
    expected_keys = ['W1', 'W2', 'W3', 'b1', 'b2', 'b3']  # 预期的权重键
    for key in expected_keys:
        if key not in network:
            raise ValueError(f"权重字典缺少关键键: {key}")
    
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

imgs, labels = get_data()
print(imgs.shape)
print(labels.shape)
print(imgs[0].shape)
print(labels.shape)

pkl_path = "E:/Codes/sample_weight.pkl"
network = init_network_predict(pkl_path)

W1, W2, W3 = network['W1'], network['W2'], network['W3']
b1, b2, b3 = network['b1'], network['b2'], network['b3']

print(W1.shape)
print(W2.shape)
print(W3.shape)
print(b1.shape)
print(b2.shape)
print(b3.shape)


accuracy_cnt = 0
for i in range(len(imgs)):
    y = predict(network, imgs[i])
    # 获取概率列表中最大值的索引，作为预测结果
    p = np.argmax(y)
    if p == labels[i]:
        accuracy_cnt += 1
        
print("Accuracy: " + str(float(accuracy_cnt) / len(imgs)))


print("-------------- 3.6.3 -------------------")
imgs, labels = get_data()
network = init_network_predict(pkl_path)

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    img_batch = imgs[i: i+batch_size]
    predict_batch = predict(network, img_batch)
    print(predict_batch.shape)  # (100, 10)
    p = np.argmax(predict_batch, axis=1)
    print(p.shape)  # (100, )
    accuracy_cnt += np.sum(p == labels[i: i+batch_size])
    
print("Accuracy: " + str(float(accuracy_cnt) / len(imgs)))

