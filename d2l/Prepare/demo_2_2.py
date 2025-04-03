import os
import pandas as pd

import torch

BASE_DIR = "E:/Codes/d2l-zh/"

# exist_ok用于控制当目标目录已存在时是否触发异常，默认为False
# 若目录已存在，raise FileExistsError异常
# 设置为True，若目录已存在，函数静默跳过，不触发异常
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
data_file = os.path.join(BASE_DIR, 'data', 'house_tiny.csv')

with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') # 表头
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 读取csv
data = pd.read_csv(data_file)
print(data)

# 处理缺失值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)

# 将类别型特征（Alley）的缺失值视为一个独立的类别，通过独热编码将其转换为机器学习模型可以理解的数值类型
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 转为张量
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(X)
print(y)

# test
data_file = os.path.join(BASE_DIR, 'data', 'test_house_tiny.csv')

with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') # 表头
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('3,NA,122000\n')
    f.write('1,Pave,140304\n')
    f.write('2,NA,142323\n')
    
test_data = pd.read_csv(data_file)

inputs, outputs = test_data.iloc[:, 0:2], test_data.iloc[:, 2]

nan_counts = inputs.isna().sum()

max_nan_column = nan_counts.idxmax()

inputs_cleaned = inputs.drop(max_nan_column, axis=1)

print(inputs_cleaned)