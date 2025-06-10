# 4.2.1 均方误差
print("-------------- 4.2.1 ----------------")
import numpy as np

def mean_squared_error(output, label):
    return 0.5 * np.sum((output - label) ** 2)

label = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
output = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(np.array(output), np.array(label)))

output = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(mean_squared_error(np.array(output), np.array(label)))
