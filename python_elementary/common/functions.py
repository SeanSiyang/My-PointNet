import numpy as np
import sys
import os

sys.path.append(os.pardir)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    
    x = x - np.max(x)   # 溢出对策
    
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error_class_label(output, label):
    delta = 1e-7
    
    # 统一形状
    if output.ndim == 1:
        output = output.reshape(1, output.size)
        label = label.reshape(1, label.size)
    
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    # 可以理解为是将one-hot编码的标签转换为类别索引的标签
    if label.size == output.size:
        label = label.argmax(axis=1)

    batch_size = output.shape[0]
    return -np.sum(np.log(output[np.arange(batch_size), label] + delta)) / batch_size