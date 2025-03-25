#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File        : 3_16_basic.py
Author      : Sean Zhang <siyang.zhang1997@outlook.com>
Date        : 2025-03-11
Description : 
    basic_1 
        - 读取数据
        - 异常捕获
        - assert 断言
        - 一条一条读取数据
        - 按batch_size大小读取数据
        
    basic_2
        - 字符串格式化

    basic_3
        - 读取打乱
        
    basic_4
        - Dataset and Dataloader
"""

import math
import os
import random
import numpy as np

# ============================================================================= 
# basic 1
# =============================================================================

# -------------------- 读取数据与try ---------------------

# 读取数据，并将数据划分为 text 和 label
def read_data(file_path):
    
    with open(file_path, encoding="utf-8") as f:
        all_data = f.read().split('\n') # 全部读进来，并且按行划分
        
    all_text, all_label = [], []
    for data in all_data:
        data_s = data.split('\t')   # 按\t分割以后，前面是文字，后面是数字标签
        
        # 处理脏数据，比如只有文字或只有标签的情况
        if len(data_s) != 2:
            continue
        
        
        # text, label = data_s
        # all_text.append(text)
        # # 使用try来判断是否有非法数据，比如说 label 并非数字
        # try:
        #     label = int(label)  # 如果label非数字，会被捕获异常
        #     all_label.append(label)
        # except:
        #     all_text.pop()        # 频繁的增删，原因是在最前面就把text加进去了，这样容易把数据-标签对打乱顺序      
        #     print("标签报错！")
        
        # 上面代码的问题：数据量如果比较大，频繁地增删会出现问题
        
        # 处理好了再添加，如果无法转为int，就直接报错了
        text, label = data_s
        try:
            label = int(label)
            all_text.append(text)
            all_label.append(label)
        except:
            print("标签报错！")

    # -------------------- assert ---------------------
    # assert 判断后面的内容是否为 false，如果为 false，则报错，输出内容就是 assert 最后的字符串
    assert len(all_text) == len(all_label), "数据和标签长度都不一样" # 判断读取的文本和标签是否数量一致
    
    return all_text, all_label

def main1():
    txt_path = "E:/Codes/手写AI/3_16_try_dataset/data/train1.txt"
    all_text, all_label = read_data(txt_path)
    
    epoch = 10

    # -------------------- 一条一条的读取数据 ---------------------
    for e in range(epoch):
        for text, label in zip(all_text, all_label):
            print(text, label)

    # -------------------- 按batch_size大小读取数据 ---------------------
    batch_size = 2
    # 首先要计算出来要拿多少次
    batch_num = int(len(all_text) / batch_size) # 如果是向下取整，可能遗漏最后没有成batch_size规模的数据
    # 在python中，/是浮点除法，始终返回float；//是整数除法，向下取整
    # 这与C++不太一样
    print(batch_num)
    
    for e in range(epoch):
        # 按batch_num来两个两个读
        for batch_idx in range(batch_num):
            # 切片访问哪怕越界了也没事
            batch_text = all_text[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_label = all_label[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            
            print(batch_text)
            print(batch_label)
            
    print("=" * 30)
    
    # 若最后剩下的数据的数量不够batch_size，可以考虑将batch_num进行向上取整，因为是切片，不会有越界的问题
    # 要向上取整，需要调用math.ceil，需要import math
    batch_num = math.ceil(len(all_text) / batch_size)
    
    for e in range(epoch):
        for batch_idx in range(batch_num):
            batch_text = all_text[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_label = all_label[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            
            print(batch_text)
            print(batch_label)
    
    # =============================================================================
    # basic 2
    # =============================================================================

    # -------------------- 字符串格式化 ---------------------
    # 基础用法与核心特性
    # 变量直接插值：在字符串前加f或F，用{}包裹变量或表达式，运行时自动求值替换
    # 场景：快速拼接动态内容，如日志输出、用户提示信息
    name = "Alice"
    age = 30
    print(f"{name} is {age} years old.")
    
    # 支持复杂表达式：可在{}内嵌入运算、函数调用
    x, y = 10, 20
    print(f"sum: {x + y}")
    print(f"Area: {math.pi * 3 ** 2:.2f}") # 28.27
    
    # 数字格式化：通过 : 添加格式说明符，控制精度、对齐等
    price = 19.99
    num = 1000000
    # 保留两位小数
    print(f"price: {price:.2f}")
    # 千分位分隔符
    print(f"Net worth: {num:,}")
    # 科学计数法
    print(f"Distance: {2.5e8:.2e}") # 结合了科学计数法表示和精度控制的格式说明符
    
    # 对齐与填充:指定宽度和对齐方式（<左对齐，>右对齐，^居中），用字符填充空白
    name = "Bob"
    print(f"{name:>10}")    # '       Bob'（右对齐，宽度10）
    print(f"{name:*^10}")   # ***Bob****（居中，宽度10，*填充）
    
    # 条件表达式与调试
    salary = 15000
    # 条件判断
    print(f"Status: {'高收入' if salary > 20000 else '普通收入'}")  # Status: 普通收入
    # 调试模式（Python 3.8+）
    # Python 3.8+ 引入的调试语法主要指通过 f-string 的 = 操作符实现快速变量追踪和表达式求值输出，极大简化了调试时的代码书写
    a, b = 5, 3
    print(f"{a=}, {b=}")  # a=5, b=3（自动打印变量名和值） {x=} 会扩展为 x=值，直接显示变量名和当前值
    
    # 对象属性与字典访问
    # 直接调用对象的属性或字典键值：
    class User:
        def __init__(self, name, age):
            self.name = name
            self.age = age
            
    user = User("Charlie", 25)
    data = {"id": 101, "role": "admin"}
    print(f"User: {user.name}, Role: {data['role']}")  # User: Charlie, Role: admin
    
    # 多行字符串与转义
    # 使用三引号处理多行内容，转义大括号和引号
    message = f"""
    Report:
    - Name: {name}
    - Age: {age}
    - Salary: ${salary:,.2f}  # 自动换行，保留格式
    """
    # 转义大括号
    print(f"{{Hello}} {name}")  # {Hello} Alice
    # 引号嵌套
    print(f'''He said: "{name}'s salary is {salary}"''')
        
    a = 10
    b = 20.11
    c = "hello"
    d = f"{a} xxx {b} {c}"
    e = f"{a} xxx {b:.2f} {c}"
    
    # 如果要在字符串中显示{} ，需要输入两次
    d = f"{a} abc {{"
    print(d) # 10 abc {
        
    # =============================================================================
    # basic 3
    # =============================================================================
    
    # -------------------- 读取打乱 ---------------------
    # 可以考虑直接将训练集手动划分为几份，人为打乱顺序
    # 如果是这种方式在读取数据的时候，就不能提前读好了，而是在epoch的迭代里不断更换读取的数据
    # for e in range(epoch):
    #     all_text, all_label = read_data(os.path.join("data", f"train{e}.txt"))    # 提前处理好数据，放到不同文件中
    #     # 更新一下每个文件要取batch的次数
    #     batch_num = math.ceil(len(all_text) / batch_size)
    
    # 上面的方法本质是打乱数据本身
    
    # 如果数据很多，很难人为处理，这种一般用于竞赛
    # 为什么要打乱索引
    """
    打破数据的内在顺序性，从而优化模型的训练效果
    消除顺序依赖性
    如果数据本身存在特定顺序（例如按类别排列），模型可能错误地学习到“数据顺序”而非“数据特征”，导致对顺序的过拟合。
    例如，前 500 个样本全为类别 A, 后 500 个全为类别 B, 模型可能简单记忆顺序而非特征
    
    缓解数据分布偏差
    当数据集中存在类别不平衡或时间序列特征时（如某些时段数据质量差），shuffle 可确保每个批次（batch）包含多样化的样本，减少模型对特定分布的依赖
    
    优化梯度下降过程
    在非打乱数据中，相邻批次的数据可能高度相似，导致梯度更新方向单一，模型陷入局部最优；而打乱后，梯度方向更稳定，模型收敛更快
    
    防止过拟合Overfitting: 打乱数据后，模型无法通过记忆顺序或特定模式来“作弊”，必须学习数据的真实特征，从而提高泛化能力
    示例：在训练图像分类模型时，若数据按类别排序，模型可能仅根据批次顺序判断类别，而非图像内容
    
    提升训练稳定性
    每个批次的数据分布更接近整体数据分布，避免模型在训练初期因连续接收同一类别的数据而剧烈震荡（如损失函数波动大）
    
    加速模型收敛
    多样化的批次数据为优化器（如 SGD、Adam）提供更全面的梯度信息，减少冗余更新，加快收敛速度
    
    支持动态数据增强
    在数据增强（如随机裁剪、旋转）场景中，打乱顺序与增强操作结合，可进一步增加数据的随机性，提升模型鲁棒性
    
    其核心价值在于打破数据中的隐含偏差，使模型更关注数据本身的特征而非外在排列。通过合理应用，可以显著提升模型性能、稳定性和泛化能力。
    """
    
    # 下面的方法不打乱数据本身，而是通过数据构建索引，打乱索引，再利用batch_size从打乱后的索引里提取索引，根据这个索引在数据本身里去提取
    
    all_text, all_label = read_data(txt_path)
    epoch = 10
    batch_size = 2
    batch_num = math.ceil(len(all_text) / batch_size) # 一共要按batch_size提取多少次数据
    
    # 打乱数据需要先构建打乱的索引序列
    random_idx = [i for i in range(len(all_text))]
    print(random_idx)   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    random.shuffle(random_idx)
    print(random_idx)   # [8, 9, 0, 1, 4, 5, 3, 10, 7, 2, 6]
    
    # 在每轮训练时，先获取一个新的乱序索引
    for e in range(epoch):
        print(f"epoch: {e}" + "=" * 30)
        random_idx = [i for i in range(len(all_text))]
        random.shuffle(random_idx) # 新的打乱后的索引列表
        
        for batch_idx in range(batch_num):
            # 根据batch_idx去索引列表中提取需要的索引
            batch_random_i = random_idx[batch_idx * batch_size : (batch_idx+1) * batch_size]
            # 根据获得的索引在数据中直接提取数据
            batch_text = [all_text[i] for i in batch_random_i]
            

# =============================================================================
# basic 4
# =============================================================================

# 获取数据并控制读取边界
class Dataset():
    """
    一条一条的读
    没有体现batch_size和打乱数据
    """
    def __init__(self, all_text, all_label, batch_size):
        self.all_text = all_text
        self.all_label = all_label
        self.batch_size = batch_size
        self.n = 0  # 控制边界
        
    def __iter__(self):
        return self

    def __next__(self):
        self.n += 1
        if self.n > len(self.all_text):
            return None
        
        return f"hello world {self.n}"
    
class Dataset_batch_size():
    def __init__(self, all_text, all_label, batch_size):
        self.all_text = all_text
        self.all_label = all_label
        self.batch_size = batch_size
        self.cursor = 0
        self.random_idx = [i for i in range(len(self.all_text))]
        random.shuffle(self.random_idx)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cursor >= len(self.all_text):
            return None
        
        batch_random_idx = self.random_idx[self.cursor : self.batch_size + self.cursor] # 构建索引
        batch_text = [self.all_text[i] for i in batch_random_idx] # 根据构建的索引去数据中提取
        batch_label = [self.all_label[i] for i in batch_random_idx] # 根据构建的索引去标签中提取
        
        self.cursor += len(batch_text)
        
        return batch_text, batch_label
    
class Dataset_epoch():
    def __init__(self, all_text, all_label, batch_size):
        self.all_text = all_text
        self.all_label = all_label
        self.batch_size = batch_size
    """
    Python的迭代器协议要求任何可迭代对象必须实现以下两个方法：
        __iter__：返回迭代器对象（通常是自身，即return self）
        __next__：定义如何获取下一个元素，若没有更多元素需抛出StopIteration异常
        
    可迭代对象：可以使用for循环来循环迭代每一个元素
    for循环迭代 可迭代对象 是先调用 可迭代对象 的__iter__()方法，得到迭代器，然后再依次调用迭代器的__next__()方法获取元素
    可迭代对象必须有__iter__()方法，但只有一个__iter__()方法难以保证被for循环调用，只有当__iter__()方法返回一个迭代器时
    才能被for循环迭代调用
    总之，要在__iter__()方法中，保证返回一个 迭代器
    
    for循环调用可迭代对象的步骤：
        a) 调用可迭代对象的__iter__()方法，得到对应的迭代器
        b) 调用迭代器的__next__()方法，得到每一个元素，直到出现StopIteration异常抛出
        
    class MyIterable:
        def __init__(self, data: Iterable):
            self.__data = data
            
        def __iter__(self):
            # 必须提供该方法，且该方法的返回值需要是一个迭代器（生成器也属于迭代器）
            yield from self.__data
            
    if __name__ == "__main__":
        obj = MyIterable("abc")
        print(obj)
        print(iter(obj))
        for _ in obj:
            print(_)
    
    迭代器是实际上能被for调用或被list调用的对象，一般自定义的可迭代对象中的__iter__方法，返回的就是迭代器
    
    只要一个对象定义了__next__()和__iter__()方法，即使两个的方法体是空的，那么该对象也是迭代器。
    
    __next__()：返回迭代器的下一个元素
    __iter__(): 一般都是返回自己
    
    # 可迭代对象
    class MyIterable:
        def __init__(self, data: Iterable):
            self.__data = data
        
        def __iter__(self):
            return MyIterator(self.__data)
        
    #迭代器
    class MyIterator:
        def __init__(self, data: Iterable):
            self.__data = data
            self.__index = 0
        
        def __iter__(self):
            return self
        
        def __next__(self):
            try:
                value = self.__data[self.__index]
                self.__index += 1
            except IndexError:
                raise StopIteration
            return value
    
    # 另外一个例子
    def __iter__(self):
        return self # 返回本身，表示该类既是可迭代对象又是迭代器
        
    def __next__(self):
        ...
    
    当某个类同时实现了这两个方法，且__iter__返回self时，该类实例：
        满足可迭代对象的条件（因有__iter__方法）；
        同时自身就是迭代器（因有__next__方法）
    
    __iter__返回self的作用:
        简化设计：
            迭代器通常需要记录遍历状态（如游标位置）。
            若类自身直接管理状态，并通过__iter__返回自身，则无需额外创建独立的迭代器类，代码更简洁（例如生成器或文件对象）        
            Python要求迭代器必须也是可迭代对象。通过__iter__返回self，迭代器可以直接用于for循环等需要可迭代对象的场景
    """
    
    def __iter__(self):
        # 每个epoch需要重置游标
        self.cursor = 0
        self.random_idx = list(range(len(self.all_text)))
        random.shuffle(self.random_idx)
        return self
    
    def __next__(self):
        if self.cursor >= len(self.all_text):
            raise StopIteration
        
        batch_random_idx = self.random_idx[self.cursor : self.cursor + self.batch_size]
        batch_text = [self.all_text[i] for i in batch_random_idx]
        batch_label = [self.all_label[i] for i in batch_random_idx]
        
        self.cursor += len(batch_text)
        
        return batch_text, batch_label
        
        
if __name__ == "__main__":
    # 一条一条的读
    txt_path = "E:/Codes/手写AI/3_16_try_dataset/data/train1.txt"
    
    all_text, all_label = read_data(txt_path)
    
    # batch_size = 2
    # train_dataset = Dataset(all_text, all_label, batch_size)
    
    # for i in train_dataset: # 如果希望能循环train_dataset，则它的类里需要有__iter__和__next__两个魔术方法
    #     if i is not None:
    #         print(i)
    #     else:
    #         break
        
    # 按batch_size来读取，并且打乱数据
    batch_size = 2
    train_dataset_batchsize = Dataset_batch_size(all_text, all_label, batch_size)
    for i in train_dataset_batchsize:
        if i is not None:
            print(i)
        else:
            break
        
    # 添加epoch，每个epoch里random不一样
    num_epochs = 10
    batch_size = 2
    
    train_dataset = Dataset_epoch(all_text, all_label, batch_size)
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1} ===")
        
        for batch_idx, (texts, labels) in enumerate(train_dataset):
            print(f"Batch {batch_idx}")
            print("Texts:", texts)
            print("Labels:", labels)