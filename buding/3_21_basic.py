#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File        : 3_21_basic.py
Author      : Sean Zhang <siyang.zhang1997@outlook.com>
Date        : 2025-04-01
Description : 
    basic_1 
        - __iter__ 从哪获取数据
        - __next__ 怎么取数据
        
    basic_2
        - 给每个文字编码
        - __getitem__
        - raise
        - 边界处理

    basic_3
        - HW：补全+shuffle逻辑
        
    basic_4
        - 每个文字编码
        - 每个标签编码
        - enumerate
    
    basic_5
        - 改成HW版本：dataloader和dataset是分离的 且 获取的数据是已经文本转编码了
        
    basic_6
        - 填充和裁剪
        - 在将文字编码的时候，索引为0的字符要做特殊处理，因为填充处理使用的是0，否则会文字的编码冲突
        - 转为numpy
"""

import random
import numpy as np

# ============================================================================= 
# basic 1
# =============================================================================

def read_data(file_path):
    """
    读取数据
    """
    with open(file_path, encoding="utf-8") as f:
        all_data = f.read().split('\n') # 全部读进来，并且按行划分
        
    all_text, all_label = [], []
    for data in all_data:
        data_s = data.split('\t')   # 按\t分割以后，前面是文字，后面是数字标签
        
        # 处理脏数据，比如只有文字或只有标签的情况
        if len(data_s) != 2:
            continue
        
        # 处理好了再添加，如果无法转为int，就直接报错了
        text, label = data_s
        try:
            all_text.append(text)
            all_label.append(label)
        except:
            print("标签报错！")
            
    assert len(all_text) == len(all_label), "数据和标签长度都不一样" # 判断读取的文本和标签是否数量一致
    
    return all_text, all_label

"""
__iter__ 在for循环时被触发，然后就会去寻找 __next__ 方法，如果返回 self，而 self 有这个方法，则会调用 __next__ 方法
如果这个类里既有__iter__又有__next__，这个类既是迭代器也是可迭代对象

如果当前类没有__next__方法，则会去__iter__的返回值中寻找是否有__next__方法，否则报错
"""
class Dataset():
    def __init__(self, all_text, all_label, batch_size):
        self.all_text = all_text
        self.all_label = all_label
        self.batch_size = batch_size
    
    # __iter__方法在循环开始仅执行一次，后续的每次迭代都调用迭代器对象的__next__方法
    def __iter__(self):
        # print("Hello iter")
        dataloader = Dataloader(self)
        
        return dataloader
        # return self
    
    # 如果类里自带__next__，然后在__iter__中返回self，则会调用__next__方法，每循环一次调用一次
    # def __next__(self):
    #     print("yes")
        
class Dataloader():
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.cursor = 0
        self.random_idx = [i for i in range(len(self.dataset.all_text))]
        random.shuffle(self.random_idx)
        
    def __next__(self):
        if self.cursor >= len(self.dataset.all_text):
            raise StopIteration
        
        # batch_i = self.random_idx[self.cursor: min(self.cursor + self.dataset.batch_size, len(self.dataset.all_text))]
        # 用切片的方式取索引，避免了在后续步骤出现的越界问题
        batch_i = self.random_idx[self.cursor: self.cursor + self.dataset.batch_size]
        
        # 根据打乱后的索引直接从数据本身拿
        batch_text = [self.dataset.all_text[i] for i in batch_i]
        batch_label = [self.dataset.all_label[i] for i in batch_i]
        
        self.cursor += self.dataset.batch_size
        
        return batch_text, batch_label
        

def main_1():
    # 对应dataset_dataloader2.py
    txt_path = "E:/Codes/手写AI/3_21_dataset_dataloader/data/train0.txt"
    all_text, all_label = read_data(txt_path)
    
    epoch = 10
    batch_size = 2
    
    # print(all_text)
    # print(all_label)
    train_dataset = Dataset(all_text, all_label, batch_size)
    for e in range(epoch):
        for i in train_dataset:
            print(i)
        else:
            break

# ============================================================================= 
# basic 2
# =============================================================================

# 将文本编码为数字
def build_word_2_index(all_text):
    """
    all_text里每个元素是一个字符串，而编码需要对每个文字进行编码，所以在取出一个text以后，还要再次遍历取出文字
    因此，可以把all_text看做是一个二级list
    """
    word_2_index = {}
    for text in all_text:
        for w in text:
            word_2_index[w] = word_2_index.get(w, len(word_2_index))
            
    return word_2_index

class Dataset2():
    def __init__(self, all_text, all_label, batch_size):
        self.all_text = all_text
        self.all_label = all_label
        self.batch_size = batch_size
    
    def __iter__(self):
        dataloader = Dataloader2(self) # 在这里会更新cursor为 0，所以每个epoch都会执行
        return dataloader
    
    def __getitem__(self, index):
        """
        getitem只实现拿一条数据的逻辑
        """
        text = self.all_text[index]
        label = self.all_label[index]
        
        return text, label

class Dataloader2():
    def __init__(self, dataset: Dataset2):
        self.dataset = dataset
        self.cursor = 0
        self.random_idx = [i for i in range(len(self.dataset.all_text))]
        random.shuffle(self.random_idx)
        
    def __next__(self):
        if self.cursor >= len(self.dataset.all_text):
            raise StopIteration
        
        # 老师的写法：本质还是在处理索引，然后根据索引去提数据
        # 这里触发了魔法方法 __getitem__
        # 因为没有使用切片处理索引，所以这里容易出现越界问题
        # batch_data = [self.dataset[i] for i in range(self.cursor, self.cursor + self.dataset.batch_size)]
        
        # 改进：使用min控制边界
        batch_data = [self.dataset[i] for i in range(self.cursor, min(self.cursor + self.dataset.batch_size, len(self.dataset.all_text)))]
        
        text, label = zip(*batch_data)
        
        self.cursor += self.dataset.batch_size
        
        return text, label

def main_2():
    # 对应dataset_dataloader3和dataset_dataloader4.py
    # dataset_dataloader3.py 新增了字符编码
    txt_path = "E:/Codes/手写AI/3_21_dataset_dataloader/data/train0.txt"
    all_text, all_label = read_data(txt_path)
    
    epoch = 10
    batch_size = 2

    word_2_index = build_word_2_index(all_text) # 将每个文字编码，不会重复编码
    
    train_dataset = Dataset2(all_text, all_label, batch_size)
    for e in range(epoch):
        for batch_text, batch_label in train_dataset: # 这里解包处理需保证__next__方法中当cursor越界后的返回值不回引起报错，如果返回的是None需要判断一下
            print(batch_text)
        else:
            break 
        
    # 如果返回的是None
    for e in range(epoch):
        for batch_data in train_dataset:
            if batch_data: # 处理 None
                batch_text, batch_label = batch_data
                print(batch_text)

    # 在 dataset_dataloader5.py 中主要解决的是next的返回值，使用了 raise StopIteration 去结束循环


# ============================================================================= 
# basic 3
# =============================================================================

# 作业：补全 dataset_dataloader6.py
class Dataset3():
    def __init__(self, all_text, all_label):
        self.all_text = all_text
        self.all_label = all_label
    
    def __getitem__(self, index):
        text = self.all_text[index]
        label = self.all_label[index]
        
        return text, label
    
class DataLoader3():
    def __init__(self, dataset: Dataset3, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.cursor = 0
        self.shuffle = shuffle
        self.random_idx = []
    
    def __iter__(self):
        # 每个epoch打乱一次，所以每个epoch在启动dataloader的迭代的时候，重新打乱
        if self.shuffle:
            self.random_idx = [i for i in range(len(self.dataset.all_text))]
            random.shuffle(self.random_idx)
        
        self.cursor = 0 # 更新游标，之前的实现因为在dataset类里实例化了dataloader对象，所以每次会更新cursor为0，此处需要手动更新
        
        return self
    
    def __next__(self):
        if self.cursor >= len(self.dataset.all_text):
            raise StopIteration
        
        if self.shuffle:
            # 根据cursor构建乱序索引
            random_batch_idx = self.random_idx[self.cursor: self.cursor + self.batch_size]
            batch_data = [self.dataset[i] for i in random_batch_idx]
        else:
            batch_data = [self.dataset[i] for i in range(self.cursor, min(self.cursor+self.batch_size, len(self.dataset.all_text)))]
        
        if batch_data:
            text, label = zip(*batch_data)
            
            self.cursor += self.batch_size
            
            return text, label
        else:
            raise StopIteration

def main_3():
    # HW: dataset_dataloader6.py
    txt_path = "E:/Codes/手写AI/3_21_dataset_dataloader/data/train0.txt"
    all_text, all_label = read_data(txt_path)
    
    epoch = 10
    batch_size = 2
    
    word_2_index = build_word_2_index(all_text)
    
    dataset = Dataset3(all_text, all_label)
    dataloader = DataLoader3(dataset, batch_size, shuffle=True)
    
    # 每个epoch打乱一次数据
    for e in range(epoch):
        for batch_data in dataloader:
            batch_text, batch_label = batch_data
            print(batch_text)
            print(batch_label)

# ============================================================================= 
# basic 4
# =============================================================================

def build_label_2_index(all_label):
    """
    编码的时候并不是给每个文字编码，而是把all_label里面的每个元素进行编码，all_label可以看做是一级list
    """
    return {k: i for i, k in enumerate(set(all_label), start=0)} # set用于去重复label

"""
enumerate

参数：start控制从哪个值开始编码
list1 = ["a", "1", "2", 5, 6, 3.4]

for i in list1:
    print(i)

for i in range(0, len(list1)):
    print(list1[i])

for i, value in enumerate(list1, start=xxx):
    print(i, value)
"""

class Dataset4():
    def __init__(self, all_text, all_label, batch_size, word_2_index, label_2_index):
        self.all_text = all_text
        self.all_label = all_label
        self.batch_size = batch_size
        self.word_2_index = word_2_index
        self.label_2_index = label_2_index
    
    def __iter__(self):
        dataloader = DataLoader4(self)
        return dataloader
    
    def __getitem__(self, index):
        text = self.all_text[index]
        label = self.all_label[index]
        
        # 将文本转为数值
        text_idx = [self.word_2_index[w] for w in text] # text为每个字编码了，所以需要列表推导式
        label_idx = self.label_2_index[label]           # label不是按单独一个字来编码的，而是给一个列表元素作为独立单位来编码，所以直接根据键获取就行
        
        return text_idx, label_idx

class DataLoader4():
    def __init__(self, dataset: Dataset4):
        self.dataset = dataset
        self.cursor = 0

    def __next__(self):
        if self.cursor >= len(self.dataset.all_text):
            raise StopIteration
        
        batch_data = [self.dataset[i] for i in range(self.cursor, min(self.cursor + self.dataset.batch_size, len(self.dataset.all_text)))]
        
        text, label = zip(*batch_data)
        
        self.cursor += self.dataset.batch_size
        
        return text, label

def main_4():
    txt_path = "E:/Codes/手写AI/3_21_dataset_dataloader/data/train0.txt"
    all_text, all_label = read_data(txt_path)
    
    epoch = 10
    batch_size = 2

    word_2_index = build_word_2_index(all_text)
    label_2_index = build_label_2_index(all_label)
    
    dataset = Dataset4(all_text, all_label, batch_size, word_2_index, label_2_index)
    
    for e in range(epoch):
        for batch_data in dataset:
            if batch_data:
                batch_text_idx, batch_label_idx = batch_data
                print(batch_text_idx)
                print(batch_label_idx)
            else:
                break

# ============================================================================= 
# basic 5
# =============================================================================

# dataset类只负责收集数据以及如何根据索引访问一条数据的逻辑
class Dataset5():
    def __init__(self, all_text, all_label, word_2_index, label_2_index):
        self.all_text = all_text
        self.all_label = all_label
        self.word_2_index = word_2_index
        self.label_2_index = label_2_index

    # 根据索引访问一个样本的逻辑
    def __getitem__(self, index): 
        text = self.all_text[index]     # self.all_text是一个二级列表，每个元素是一个字符串，使用index取出来了一行文本
        label = self.all_label[index]   # self.label是一个一级列表，每个元素虽然是字符串，但是处理的时候是当做单独的一个元素来处理，不需要再去处理每个文字了
        
        text_idx = [self.word_2_index[w] for w in text] # 使用for循环遍历字符串，获取每个文字，再通过文字去word_2_index中获取对应的编码
        label_idx = self.label_2_index[label]           # 编码的时候是把里面单独的一个元素拿来编码的，所以不需要列表推导式，直接根据label值获取对应的编码
        
        return text_idx, label_idx

class DataLoader5():
    def __init__(self, dataset: Dataset5, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_idx = []
    
    def __iter__(self):
        self.cursor = 0
        if self.shuffle:
            self.random_idx = [i for i in range(len(self.dataset.all_text))]
            random.shuffle(self.random_idx)
        
        return self
    
    def __next__(self):
        if self.cursor >= len(self.dataset.all_text):
            raise StopIteration
        
        if self.shuffle:
            # 根据cursor构建索引，再去获取
            batch_idx = self.random_idx[self.cursor: self.cursor+self.batch_size]
            batch_data = [self.dataset[i] for i in batch_idx]
        else:
            batch_data = [self.dataset[i] for i in range(self.cursor, min(self.cursor+self.batch_size, len(self.dataset.all_text)))]
        
        if batch_data:
            text, label = zip(*batch_data)
            self.cursor += self.batch_size
        else:
            raise StopIteration
        

        
        return text, label

def main_4():
    txt_path = "E:/Codes/手写AI/3_21_dataset_dataloader/data/train0.txt"
    all_text, all_label = read_data(txt_path)
    
    epoch = 10
    batch_size = 2

    word_2_index = build_word_2_index(all_text)
    label_2_index = build_label_2_index(all_label)
    
    train_dataset = Dataset5(all_text, all_label, word_2_index, label_2_index)
    train_dataloader = DataLoader5(train_dataset, batch_size, shuffle=True)
    
    for e in range(epoch):
        for batch_data in train_dataloader:
            text, label = batch_data
            print(text)
            print(label)

# ============================================================================= 
# basic 6
# =============================================================================

# 这个版本主要是处理截断和填充，这个操作主要是在Dataset准备数据阶段就处理好（__getitem__）
# 转numpy操作：np.array()

def build_word_2_index_fix(all_text):
    word_2_index = {"PAD": 0}
    
    for text in all_text:
        for w in text:
            word_2_index[w] = word_2_index.get(w, len(word_2_index))
            
    return word_2_index

class Dataset6():
    def __init__(self, all_text, all_label, word_2_index, label_2_index, max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.word_2_index = word_2_index
        self.label_2_index = label_2_index
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.all_text[index][:self.max_len] # 使用max_len截断
        label = self.all_label[index]
        
        text_idx = [self.word_2_index[w] for w in text]
        # 如果长度不够，用0补充
        text_idx_p = text_idx + [0] * (self.max_len - len(text_idx)) # 在后面填充
        label_idx = self.label_2_index[label]          
        
        return text_idx_p, label_idx
    
class DataLoader6():
    
    def __init__(self, dataset: Dataset6, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_idx = []
    
    def __iter__(self):
        self.cursor = 0
        if self.shuffle:
            self.random_idx = [i for i in range(len(self.dataset.all_text))]
            random.shuffle(self.random_idx)
        return self
    
    def __next__(self):
        if self.cursor >= len(self.dataset.all_text):
            raise StopIteration
        
        if self.shuffle:
            # 根据cursor去打乱的索引中构建需要的索引，再利用新的索引去获取数据
            batch_idx = self.random_idx[self.cursor: self.cursor+self.batch_size]
            batch_data = [self.dataset[i] for i in batch_idx]
        else:
            batch_data = [self.dataset[i] for i in range(self.cursor, min(self.cursor+self.batch_size, len(self.dataset.all_text)))]
        
        if batch_data:
            text, label = zip(*batch_data)
            self.cursor += self.batch_size
            return np.array(text), np.array(label) # 转为numpy
        else:
            raise StopIteration
    
def main_5():
    # 对应dataset_dataloader8.py
    txt_path = "E:/Codes/手写AI/3_21_dataset_dataloader/data/train0.txt"
    all_text, all_label = read_data(txt_path)
    
    epoch = 10
    batch_size = 2
    max_len = 10 # 统一字符长度，在__get_item__控制

    word_2_index = build_word_2_index_fix(all_text)
    label_2_index = build_label_2_index(all_label)
    
    train_dataset = Dataset6(all_text, all_label, word_2_index, label_2_index, max_len)
    train_dataloader = DataLoader6(train_dataset, batch_size, shuffle=True)
    
    for e in range(epoch):
        for batch_data in train_dataloader:
            if batch_data:
                text, label = batch_data
                print(text.shape)
                print(label.shape)

if __name__ == "__main__":
    main_5()