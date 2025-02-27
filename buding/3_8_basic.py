#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File        : 3_8_basic.py
Author      : Sean Zhang <siyang.zhang1997@outlook.com>
Date        : 2025-02-26
Description : 
    basic_1 
        - 匿名函数
        - eval
        - 可迭代对象
        - 魔术方法
        - dir 函数
        - 列表推导式
        
    basic_2
        - 字典
        - 字典推导式
        - 迭代器、生成器
    
    basic_3
        - zip
        - sort
        
    basic_4
        - sorted
        - sorted 与 dict
    
    basic_5
        - 文件读取
"""

# =============================================================================
# basic 1
# =============================================================================

# -------------------- 匿名函数 ---------------------
y = lambda x: x + 1     # y = x + 1
y2 = lambda x: x ** 2   # lambda后面的值就是参数，冒号后面是具体操作

print(y(10))    # 调用的时候，就把左值当做匿名函数的名字来使用
print(y2(100))  # 10000

# -------------------- eval ---------------------
# 作用：将字符串解析为有效的Python表达式并执行
# eval(expression, globals=None, locals=None)
# globals(option)：全局变量的字典
# locals(option)：局部变量的字典
x = 0
y = eval("x + 1")
print(y)    # 1

str1 = "{'a' : 1, 'b' : 2}" # 将字符串中的字典使用eval提取出来
d = eval(str1)
print(d)            # {'a': 1, 'b': 2}
print(type(d))      # <class 'dict'>

# -------------------- 可迭代对象 ---------------------
# 可迭代对象必须是一个可以逐个返回元素的容器（如列表、元组、字符串、字典、集合、生成器等）
# for i in 10:    # TypeError: 'int' object is not iterable
#     print(i)    # 整数（int） 是单个值，无法被迭代。
# 不可迭代对象：整数、浮点数、布尔值等单个值。
for i in range(10):
    print(i)
    
# -------------------- 魔术方法 ---------------------
# 魔术方法在特定场景下自动触发
# 想要被迭代，需要有__iter__函数，如果自定义类，且希望对象是一个可迭代对象，需要实现两个函数
# __iter__ 和 __next__ 方法
# __iter__() 方法返回一个迭代器对象。该方法在循环开始时被调用，通常它返回一个实现了 __next__() 方法的对象
# __next__() 方法用于返回容器中的下一个元素。当没有更多元素时，应该抛出 StopIteration 异常，表示迭代已经结束
class MyRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.current = start # 当前迭代位置
    
    # 实现 __iter__ 方法，返回迭代器本身
    def __iter__(self):
        return self # self是迭代器
    
    # 实现 __next__ 方法，定义如何迭代
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration # 当迭代到终点时抛出 StopIteration 异常
        self.current += 1
        return self.current - 1 # 返回当前值
    
my_range = MyRange(0, 5)
for num in my_range:
    print(num)

# -------------------- dir 函数 ---------------------
# dir函数返回对象的有效属性列表，包括方法和属性
# 通过callable()判断是否为方法
a = 10
for attr in dir(a):
    if callable(getattr(a, attr)):
        print(f"{attr} 是一个方法")
    else:
        print(f"{attr} 是成员属性")

# 遍历数字的每一位
number = 123
for digit_str in str(number):
    digit = int(digit_str)
    print(digit)
    
# -------------------- 列表推导式 ---------------------
list1 = [10, 30, 4, 5, 7, 6, 43, 109]

print(list1)    # [10, 30, 4, 5, 7, 6, 43, 109]
for i in list1:
    i += 1          # 这样不会修改list1的值，i只是副本
print(list1)    # [10, 30, 4, 5, 7, 6, 43, 109]

# 如果希望修改列表中的元素，需要使用下标访问
for i in range(len(list1)):
    list1[i] += 1
print(list1)    # [11, 31, 5, 6, 8, 7, 44, 110]

# 但是修改原列表的形式，如果在修改过程中改变了列表的长度，会导致列表越界的问题
list1 = [10, 30, 4, 5, 7, 6, 43, 109]
# 把列表中的偶数加1，将奇数删除
# for i in range(len(list1)):
#     if list1[i] % 2 == 0:
#         list1[i] += 1
#     else:
#         list1.pop(i) # IndexError: list index out of range
# 报错的原因是改变了原列表的长度，导致index发生了变化
# 最好的解决办法是不要操作原列表，将想要的元素放到一个新的列表中

list1 = [10, 30, 4, 5, 7, 6, 43, 109]
list2 = []

for i in range(len(list1)):
    if list1[i] % 2 == 0:
        list2.append(list1[i] + 1)

list3 = []
for i in list1:
    if i % 2 == 0:
        list3.append(i + 1)

# 使用列表推导式直接解决
list3 = [i + 1 for i in list1 if i % 2 == 0]

# =============================================================================
# basic 2
# =============================================================================

# -------------------- 字典 ---------------------
dict1 = {"a": 1, "b": 2, "c": 3}
for i in dict1:
    print(i)    # 键

for i in dict1.values():
    print(i)    # 值

for i in dict1.items():
    print(i)    # (键, 值) ('a', 1)
    
for i, j in dict1.items():
    print(i, j) # a 1
    
i = 10
# list(i) # TypeError: 'int' object is not iterable 

# 必须得是可迭代对象才可以被list强转
a = list(dict1.items())

# -------------------- 字典推导式 ---------------------
for i, j in dict1.items():
    dict1[i] = j + 1
print(dict1)

# 字典推导式
dict2 = {i: j + 1 for i, j in dict1.items()}
print(dict2)

dict3 = {k: dict1[k] + 1 for k in dict1}
print(dict3)

# 可迭代对象（Iterable）、迭代器（Iterator）和生成器（Generator）
# -------------------- 可迭代对象 ---------------------
# 可迭代对象：任何可以通过 for 循环遍历的对象
# 核心特征：必须实现 __iter__() 方法，返回一个 迭代器（Iterator）
# 常见例子：列表、元组、字符串、字典、集合、文件对象等

from collections.abc import Iterable

print(isinstance([1, 2, 3], Iterable))  # True，列表是可迭代对象
print(isinstance(123, Iterable))        # False，整数不可迭代

# -------------------- 迭代器 ---------------------
# 迭代器：负责逐个返回元素的对象
# 核心特征：
#   必须实现 __iter__() 方法（返回自身）
#   必须实现 __next__() 方法（返回下一个元素，无元素时抛出 StopIteration 异常）
# 特点：迭代器是“一次性”的，遍历结束后无法重复使用

from collections.abc import Iterator

my_list = [1, 2, 3]
list_iterator = iter(my_list)

print(isinstance(list_iterator, Iterator))     # True
print(isinstance(my_list, Iterator))            # False, 列表本身不是迭代器

# 手动遍历迭代器
my_list = [1, 2, 3]
iterator = iter(my_list)

print(next(iterator))
print(next(iterator))
print(next(iterator))
# print(next(iterator))   # StopIteration

# -------------------- 生成器 ---------------------
# 生成器：一种特殊的迭代器，通过 生成器函数（含 yield 语句的函数）或 生成器表达式 创建。
# 核心特征：
#   按需生成值（惰性计算），节省内存
#   自动实现 __iter__() 和 __next__() 方法

def count_up_to(n):
    current = 1
    while current <= n:
        yield current   # 每次 yield 返回一个值，暂停执行
        current += 1

gen = count_up_to(3)
print(next(gen))      # 1
print(next(gen))      # 2
print(next(gen))      # 3
# print(next(gen))      # 抛出 StopIteration

# 生成器表达式
gen = (x ** 2 for x in range(3))
print(list(gen))

# 可迭代对象通过iter函数可以转换为迭代器，生成器是一种特殊的迭代器，按需生成值
# 可迭代对象 → 通过 iter() → 迭代器 → 通过 next() → 逐个元素
# 生成器 直接是迭代器，无需转换

# 可迭代对象（列表）
my_list = [1, 2, 3]

# 转换为迭代器
iterator = iter(my_list)

# 遍历
for num in iterator:
    print(num)  # 1, 2, 3

# 生成器函数
def square_numbers(nums):
    for n in nums:
        yield n**2

# 生成器是迭代器，可直接遍历
gen = square_numbers([1, 2, 3])
for num in gen:
    print(num)  # 1, 4, 9
"""
可迭代对象 vs 迭代器：
    可迭代对象（如列表）可以多次遍历（每次遍历生成新的迭代器）。
    迭代器只能遍历一次，遍历完状态失效。

迭代器 vs 生成器：
    所有生成器都是迭代器。
    生成器通过 yield 简化迭代器的实现。
    
for语句的in后面可以是迭代器也可以是可迭代对象
"""

class MyClass:
    def __init__(self, data):
        self.data = data
    
    # 实现 __iter__()，返回一个迭代器
    def __iter__(self):
        # 使用生成器函数逐个返回元素
        for item in self.data:
            yield item

# 使用示例
my_obj = MyClass([1, 2, 3])
my_list = list(my_obj) 

class MyCustomRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __getitem__(self, index):
        if index >= (self.end - self.start + 1):
            raise IndexError
        return self.start + index

# 转换为列表
my_obj = MyCustomRange(1, 3)
print(list(my_obj))  # 输出 [1, 2, 3]

# 传统实现
class MyIterator:
    def __init__(self, data):
        self.data = data        # 数据源
        self.index = 0          # 跟踪当前遍历位置

    # 迭代器必须实现 __iter__，返回自身
    def __iter__(self):
        return self

    # 实现 __next__，返回下一个元素或抛出 StopIteration
    def __next__(self):
        if self.index < len(self.data):
            value = self.data[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration  # 遍历结束时抛出异常

class MyClass1:
    def __init__(self, data):
        self.data = data  # 存储数据

    # __iter__ 返回一个迭代器对象
    def __iter__(self):
        return MyIterator(self.data)  # 返回自定义迭代器实例
    
# 创建可迭代对象
my_obj = MyClass([1, 2, 3])

# 转换为列表
print(list(my_obj))  # 输出 [1, 2, 3]

# 手动遍历
iterator = iter(my_obj)
print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3
# print(next(iterator))  # 抛出 StopIteration

# =============================================================================
# basic 3
# =============================================================================

# -------------------- zip ---------------------

# 同时循环两个列表
list1 = [1, 3, 5, 5, 7]
list2 = [5, 6, 7, 9, 8]

for i in range(5):
    print(list1[i], list2[i])

for i in zip(list1, list2):
    print(i)    # (1, 5)

for i, j in zip(list1, list2):
    print(i, j)

list1 = [1, 3, 5, 5, 7]
list2 = [5, 6, 7]
list3 = [6, 7, 8, 4, 5]

for i, j, k in zip(list1, list2, list3):
    print(i, j ,k)  # 按最短的来，可以方便统一文本长度

# -------------------- sort ---------------------
# 常见 error
list4 = list3.sort() # 因为sort的返回值为None，所以无法迭代list4
# for i in list4: # TypeError: 'NoneType' object is not iterable
#     print(i)

# 给元组排序，虽然元组的值不能修改，但可以转为list修改好以后，再转回tuple
t = (3, 4, 56, 2, 4, 65)
# print(t.sort()) # AttributeError: 'tuple' object has no attribute 'sort'
print(list(t).sort()) # sort函数的返回值为 None
# 上面的排序不会影响 t，因为list(t)返回的是一个临时变量，这个临时变量没有变量接收，排完序就没了

# 正确处理方式：转为list的过程应有个变量接收
a = list(t)
a.sort()
t1 = tuple(a)
print(t1)   # (2, 3, 4, 4, 56, 65)

# =============================================================================
# basic 4
# =============================================================================

# -------------------- sorted ---------------------

# sorted函数有返回值，此外reverse参数可以控制排序方向，key可以控制排序方式
list1 = [5, 8, 7, -20] 
print(sorted(list1))                # [-20, 5, 7, 8]
print(sorted(list1, reverse=True))  # [8, 7, 5, -20]

# 倒序的另外一种方式
print(sorted(list1))                # [-20, 5, 7, 8]
list1 = sorted(list1)
list2 = list1[::-1] 
print(list2)                        # [8, 7, 5, -20]

# 如果list嵌套有其他类型，则以里面元素的第一个值来排序，第一个相同就比较第二个，若某个位置没有值则当做 负无穷 处理
list1 = [[3, 4], [3, 4, -1]]                # 第一个元素的第三个值被视为负无穷
print(sorted(list1))                        # [[3, 4], [3, 4, -1]]

list2 = [[5, 8, 7, -20], [3, 5], [3, 5, 4]]
print(sorted(list2))                        # [[3, 5], [3, 5, 4], [5, 8, 7, -20]]

# 不同嵌套类型，经过sorted函数以后，最外层都是列表
list2 = [(5, 8, 7, -20), (3, 5), (3, 5, 4)]
print(sorted(list2))                        # [(3, 5), (3, 5, 4), (5, 8, 7, -20)]

list2 = ((5, 8, 7, -20), (3, 5), (3, 5, 4))
print(sorted(list2))                        # [(3, 5), (3, 5, 4), (5, 8, 7, -20)]

list2 = ([5, 8, 7, -20], [3, 5], [3, 5, 4])
print(sorted(list2))                        # [[3, 5], [3, 5, 4], [5, 8, 7, -20]]

# 以子元素中的总和大小作为排序依据，需要定义 key，可以使用匿名函数lambda来定义key
list3 = [[5, 6, 7, 8], [3, 4, -1], [3, 4]]
y = lambda x: sum(x)
list4 = sorted(list3, key=y)
list5 = sorted(list3, key=y, reverse=True)
list6 = sorted(list3, key=lambda x: sum(x)) # x 就是list3中的每个元素，遍历出来以后放入冒号后面的操作中

print(list4) # [[3, 4, -1], [3, 4], [5, 6, 7, 8]]
print(list5) # [[5, 6, 7, 8], [3, 4], [3, 4, -1]]
print(list6) # [[3, 4, -1], [3, 4], [5, 6, 7, 8]]

# 按最后一个元素来排序
list3 = [[5, 6, 7, 8], [3, 4, -1], [3, 4]]
y2 = lambda x: x[-1]
list4 = sorted(list3, key=y2)
print(list4) # [[3, 4, -1], [3, 4], [5, 6, 7, 8]]

# -------------------- sorted 与 dict ---------------------

d1 = {"abc": 1, "def": 2, "ghi": 3}
print(sorted(d1, key=lambda x: d1[x]))                      # 按键排序 ['abc', 'def', 'ghi']
print(sorted(d1.items(), key=lambda x: x[1], reverse=True)) # 按值排序 [('ghi', 3), ('def', 2), ('abc', 1)]

# 需求：按tuple第一个元素长度判断大小，，若字符串长度相等，按元组第二个元素判断
# 两个判断依据可以以 元组 的形式构建 lambda
d1 = {"b": ("123", 11), "a": ("123", 10), "z": ("123", 99), "c": ("1234", 12)}
print(sorted(d1.items(), key=lambda x: (len(x[1][0]), x[1][1])))                # [('a', ('123', 10)), ('b', ('123', 11)), ('z', ('123', 99)), ('c', ('1234', 12))]
print(sorted(d1.items(), key=lambda x: (len(x[1][0]), -x[1][1])))               # [('z', ('123', 99)), ('b', ('123', 11)), ('a', ('123', 10)), ('c', ('1234', 12))]
print(sorted(d1.items(), key=lambda x: (len(x[1][0]), x[1][1]), reverse=True))  # [('c', ('1234', 12)), ('z', ('123', 99)), ('b', ('123', 11)), ('a', ('123', 10))]

# =============================================================================
# basic 5
# =============================================================================

# -------------------- 文件读取 ---------------------
data_path = "E:/Codes/手写AI/3_8_基础_匿名函数_sorted/data/text.txt"
f = open(data_path, encoding="utf-8") # 打开文件，并设置编码格式
book = f.read() 
f.seek(0)           # 再次读取之前需要将指针放到第一个位置
book1 = f.read()    # 读取文件
print(book)
print(book1)
f.close()           # 关闭文件

# with 上下文管理器，核心功能：安全地打开文件并自动释放资源
with open(data_path, encoding="utf-8") as f:    # 打开文件并绑定到变量 f
    f.read()                                    # 读取文件内容
# 在代码块执行完毕后，无论是否发生异常，文件都会自动关闭（无需手动调用 f.close()）

# 传统实现
f = open(data_path, encoding="utf-8")
try:
    f.read()
finally:
    f.close()