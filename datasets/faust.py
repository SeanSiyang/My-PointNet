import os
import sys
import numpy as np

from torch.utils.data import Dataset

# 确保导包顺利
ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../')  # 将绝对路径添加到 ROOT_DIR
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    
class ShapeDataset(Dataset):
    TRAIN_IDX = np.arange(0, 80)
    TEST_IDX = np.arange(80, 100)
    
    
    pass