import numpy as np
import sys
import os

sys.path.append(os.pardir)

def _numerical_gradient_1d(function, weight):
    """
    计算一维数组输入的函数数值梯度
    
    参数:
        function: 目标函数，接受一维数组输入，返回标量值
        weight: 一维输入数组，权重
        
    返回:
        grad: 梯度向量，形状与x相同
    """
    h = 1e-4
    grad = np.zeros_like(weight)
    
    for idx in range(weight.size):
        tmp_val = weight[idx]
        
        # f(x+h)
        weight[idx] = float(tmp_val) + h
        fxh1 = function(weight)
        
        # f(x-h)
        weight[idx] = float(tmp_val) - h
        fxh2 = function(weight)
        
        # 使用中心差分法计算偏导数
        grad[idx] = (fxh1 - fxh2) / 2 * h
        
        # 恢复原始值，避免修改原数组
        weight[idx] = tmp_val
    
    return grad

def numerical_gradient_2d(function, weight):
    """
    计算二维数组输入的数值梯度（支持批量处理）
    
    参数:
        function: 目标函数，接受一维数组输入，返回标量值
        weight: 二维输入数组（批次大小, 特征维度）
        
    返回:
        grad: 梯度矩阵，形状与X相同
    """
    if weight.ndim == 1:
        return _numerical_gradient_1d(function, weight)
    else:
        grad = np.zeros_like(weight)
        
        # 遍历每一行（每个样本）
        for idx, x in enumerate(weight):
            # 对每个样本计算梯度
            grad[idx] = _numerical_gradient_1d(function, x)
        
        return grad

def numerical_gradient_official(function, weight):
    """
    通用数值梯度计算函数，支持任意维度的输入
    
    参数:
        function: 目标函数，接受与x相同形状的输入，返回标量值
        weight: 任意维度的输入数组
        
    返回:
        grad: 梯度数组，形状与x相同
    """
    h = 1e-4
    grad = np.zeros_like(weight)
    
    # 创建多维迭代器，用于遍历任意维度的数组
    # flags=['multi_index'] 允许获取元素的多维索引
    # op_flags=['readwrite'] 允许读写数组元素
    it = np.nditer(weight, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        # 获取当前元素的多维索引
        idx = it.multi_index
        tmp_val = weight[idx]  # 保存当前位置的原始值
        
        # 计算f(x+h)
        weight[idx] = float(tmp_val) + h  # 将当前元素增加h
        fxh1 = function(weight)  # 计算函数值
        
        # 计算f(x-h)
        weight[idx] = tmp_val - h  # 将当前元素减少h
        fxh2 = function(weight)  # 计算函数值
        
        # 使用中心差分法计算偏导数
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        
        # 恢复原始值
        weight[idx] = tmp_val
        
        # 移动到下一个元素
        it.iternext()
    
    return grad