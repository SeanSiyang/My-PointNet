import numpy as np
import glob
import os
import sys

# E:\Codes\Github\My-PointNet\data_utils
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在目录
# E:\Codes\Github\My-PointNet
ROOT_DIR = os.path.dirname(BASE_DIR) # 获取项目根目录
# print(BASE_DIR)
# print(ROOT_DIR)

"""
sys.path: 是一个 Python 列表，包含了 Python 在运行时搜索模块的路径。
当你在代码中 import 一个模块时, Python 会按照 sys.path 中的路径顺序去查找这个模块。
"""
sys.path.append(BASE_DIR)  # 将BASE_DIR添加到Python的模块搜索路径中

# DATA_PATH = os.path.join(ROOT_DIR, 'data', 's3dis', 'Stanford3dDataset_v1.2_Aligned_Version')
DATA_PATH = "E:\datasets\S3DIS\Stanford3dDataset_v1.2_Aligned_Version"

g_classes = [x.rstrip() for x in open(os.path.join(BASE_DIR, 'meta/class_names.txt'))]


# -----------------------------------------------------------------------------
# CONVERT ORIGINAL DATA TO OUR DATA_LABEL FILES
# -----------------------------------------------------------------------------

def collect_point_label(anno_path, out_filename, file_format='txt'):
    """ Convert original dataset files to data_label file (each line is XYZRGBL).
        We aggregated all the points from each instance in the room. 汇总了房间里每个实例的所有点 

    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save collected points and labels (each line is XYZRGBL)
        file_format: txt or numpy, determines what file format to save.
    Returns:
        None
    Note:
        the points are shifted before save, the most negative point is now at origin.
    """
    print(anno_path)
    print(out_filename)
    # points_list = []
    
    
    # for f in glob.glob(os.path.join(anno_path, "*.txt")):
    #     cls = os.path.basename(f).split('_')[0]
        
    
    pass

