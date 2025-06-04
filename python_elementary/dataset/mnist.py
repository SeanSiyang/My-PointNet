try:
    import urllib.request
except:
    raise ImportError("Don't use Python3.x")

import os
import gzip
import pickle
import numpy as np

url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = "E:/datasets/MNIST/mnist_python/"
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return
    
    print("[Note] Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("[Note] Done")


def download_mnist():
    for v in key_file.values():
        _download(v)


# 定义加载图像数据的函数
def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    print("[Note] Converting " + file_name + " to Numpy Array ...")
    # 用gzip打开压缩文件（MNIST数据集为gzip格式）
    with gzip.open(file_path, 'rb') as f:   # 'rb'表示二进制读取模式
        # 从文件读取数据：
        # 1. f.read() 读取全部字节
        # 2. np.frombuffer 将字节转换为uint8数组
        # 3. offset=16 跳过前16字节的文件头（MNIST图像文件头包含元数据）
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    
    # 将一维数组重塑为二维数组：每行代表一个图像（img_size=28x28=784）
    data = data.reshape(-1, img_size)
    print("[Note] Done")
    
    return data


# 定义加载标签数据的函数
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    print("[Note] Converting " + file_name + " to Numpy Array ...")
    with gzip.open(file_path, 'rb') as f:
        # 标签文件头为8字节，跳过前8字节
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("[Note] Done")
    
    return labels   # 返回一维标签数组


# 将MNIST四个数据集整合到字典中
def convert_numpy():
    dataset = {}
    # 分别加载训练/测试的图像和标签
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    # print(dataset['train_img'].shape)
    # print(dataset['train_label'].shape)
    # print(dataset['test_img'].shape)
    # print(dataset['test_label'].shape)
    
    return dataset


def init_mnist():
    download_mnist()
    dataset = convert_numpy()
    # 将处理好的数据集保存为pickle文件
    print("[Note] Creating pickle file ...")
    with open(save_file, 'wb') as f:    # 'wb'表示二进制写入
        # pickle.dump 将Python对象序列化到文件
        # 参数说明：
        #   dataset: 要保存的对象
        #   f: 文件对象
        #   -1: 使用最高协议版本（二进制格式，效率最高）
        pickle.dump(dataset, f, protocol=-1)
    print("[Note] Done")



def _change_one_hot_label(X):
    """将类别标签转换为one-hot编码格式
    
    Parameters
    ----------
    X : 包含类别标签的一维数组，每个元素表示样本的类别，默认类别为10个（0~9）
    
    Returns
    -------
    T : 二维数组，每行是对应样本的one-hot编码
    """
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        # 该操作会直接修改T数组本身，Numpy数组的切片操作返回的是视图而不是副本，对视图的修改会直接反映到原始数组上
        row[X[idx]] = 1
    
    return T

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集
    
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_mnist()
    
    # 从pickle文件加载数据集
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)        # 加载字典格式的数据
    
    # 像素归一化处理：有利于神经网络训练（避免大数值导致梯度爆炸）
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)  # 转换为浮点数
            dataset[key] /= 255.0                           # 将0-255的像素值归一化到0-1范围
        
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    
    # 图像维度处理（是否保持2D结构），flatten为True时，返回60000x784，即28x28=784，每张图像是1x784，适合全连接层
    # flatten为False时，返回4D张量，适合卷积网络运算
    # 同一份MNIST加载代码可以适应不同神经网络架构的需求
    if not flatten:
        for key in ('train_img', 'test_img'):
            # 将图像转换为4D张量：(样本数, 通道数, 高度, 宽度)
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

if __name__ == '__main__':
    # init_mnist()
    load_mnist(flatten=True, normalize=False, one_hot_label=True)