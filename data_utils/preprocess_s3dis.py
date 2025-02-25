import os
import sys

from utils import DATA_PATH, collect_point_label

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))] # txt中每行都是一个场景
# print(anno_paths)
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths] # 获得完整路径

# output_folder = os.path.join(ROOT_DIR, 'data/stanford_indoor3d')
output_folder = "E:\datasets\S3DIS\stanford_indoor3d"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for anno_path in anno_paths:
    print(anno_path)
    try:
        elements = anno_path.split('/')
        print(elements) # ['E:\\datasets\\S3DIS\\Stanford3dDataset_v1.2_Aligned_Version\\Area_6', 'office_26', 'Annotations']
        out_filename = elements[-3] + '_' + elements[-2] + '.npy'
        print(out_filename) # E:\datasets\S3DIS\Stanford3dDataset_v1.2_Aligned_Version\Area_6_pantry_1.npy
        collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
    except:
        print(anno_path, 'ERROR!')


