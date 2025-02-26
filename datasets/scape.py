import os
import sys
import numpy as np

ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../')  # 将绝对路径添加到 ROOT_DIR
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    
from datasets.faust import ShapeDataset as FaustShapeDataset # 别名处理：将导入的父类重命名为 FaustShapeDataset，从而避免与当前模块中定义的子类 ShapeDataset 重名

class ScapeShapeDataset(FaustShapeDataset):
    TRAIN_IDX = np.arange(0, 51)
    TEST_IDX = np.arange(51, 71)