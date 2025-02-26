#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File        : 3_7_basic.py
Author      : Sean Zhang <siyang.zhang1997@outlook.com>
Date        : 2025-02-24
Description : 
    basic_1 
        - 整型
        - 变量类型 type
        - 浮点数四舍五入
        - 切片与索引
        - 列表 list
        - 元组 tuple
        
    basic_2
        - 解包和zip
    
    basic_3
        - 字典的获取、新增、删除与修改
        
    basic_4
        - 集合
    
    basic_5
        for
"""

# =============================================================================
# basic 1
# =============================================================================

# -------------------- 整型和变量类型 ---------------------
val = 10  # 整型
val = 10 + 2
val = 10 ** 3  # 10的三次方
print("val = ", val)
print(type(val))  # 查看变量的类型

# -------------------- 浮点数四舍五入 ---------------------
val2 = 10.1
val3 = int(val2 + 0.5)
print(val3)

# -------------------- 字符串的单双引号 ---------------------
# 字符串可以使用双引号和单引号，如果要在里面使用同样的引号时要使用转义符
str1 = "abc"
str2 = "'abc'"
str3 = "\"abc\""
str4 = '\'abc\''

# -------------------- 切片和索引 ---------------------
# 切片
# 索引越界会报错，切片越界不会报错
# 在不知道字符串长度的情况下，不需要写逻辑去判断长度，可以直接给一个比较大的值去获得字符串
list1 = [1, 2, 3, 4, 5, 6]
# print(list1[100000]) # IndexError: list index out of range
print(list1[0:100000]) # [1, 2, 3, 4, 5, 6]

# -------------------- 切片的步长注意事项 ---------------------
# 步长不合理会没有输出
# start < end --> step > 0
# start > end --> step < 0
f = "123456"
print(f[1:5:1])
print(f[-1:-3]) # 空
print(f[-1:-3:-1])

# -------------------- 切片和索引的降维区别 ---------------------
# 切片取值不会降维，索引取值会降维
list1 = [[1, 2], [3, 4], [5, 6]]
print(list1[0]) # [1, 2] 降维了
print(list1[0:2]) # [[1, 2], [3, 4]] 不降维

# -------------------- 列表 list ---------------------
list1 = [12, 3, 4, 5, 6, "abc", 23.3, [123], ["abc1"]]
list2 = [[1,2,0], [3,4,5], [4,5,7]]
print(list2[0]) # [1, 2, 0]

# 列表支持修改某个元素的值，但是字符串不能修改其中一个字符，因为字符串是存在栈中的
list2[0] = "abcdef"
print(list2) # ['abcdef', [3, 4, 5], [4, 5, 7]]

str1 = "abc哈哈哈哈哈哈"
print(str1[0]) # a
# str1[0] = 'c' # TypeError: 'str' object does not support item assignment

# -------------------- 元组 tuple ---------------------
# 元组最大的特点是里面的元素不可以变
t1 = (3, 4, 5, 6)
print(t1) # (3, 4, 5, 6)
# t1[0] = 4 # TypeError: 'tuple' object does not support item assignment 

# 元组和一般变量
t1 = (1, 3)
t2 = (10)
t3 = (2, )
t4 = 3, 4, 5, 6

print(type(t1)) # <class 'tuple'>
print(type(t2)) # <class 'int'>
print(type(t3)) # <class 'tuple'>
print(type(t4)) # <class 'tuple'> # 

# =============================================================================
# basic 2
# =============================================================================

# -------------------- 解包和zip函数 ---------------------
list1 = [1, 2, (3, 4, [5, 6])]
j = list1
# a, b = list1 # ValueError: too many values to unpack (expected 2)
a, b, c = list1
print(a)
print(b)
print(c) # (3, 4, [5, 6])
# a, b, c, d = list1 # ValueError: not enough values to unpack (expected 4, got 3)
# a, b, c, d, e = list1 # ValueError: not enough values to unpack (expected 5, got 3)
a, b, (c, d, e) = list1
print(a) 
print(b)
print(c) # 3 
print(d) # 4
print(e) # [5, 6]
a, b, (c, d, [e, f]) = list1

list2 = [1, 2, (3, 4, 5)]
# a, b, c, d = list2 # ValueError: not enough values to unpack (expected 4, got 3)
# a, b, c, d, e = list2 # ValueError: not enough values to unpack (expected 5, got 3)
a, b, c = list2
a, b, (c, d, e) = list2
# 并不会自动解包，c直接就得到了(3, 4, 5)
# a, b, c, d, e = list2 # ValueError: not enough values to unpack (expected 5, got 3)


# * 是解包操作符
list1 = [(1,2,"a"), (3,4,"b"), (5,6,"c"), (7,8,"d")]
# 解包以后变成：*list1 ➔ (1,2,"a"), (3,4,"b"), (5,6,"c"), (7,8,"d")
# 即将4个元组作为4个参数传递给zip函数
# zip函数接收多个可迭代对象（这里是4个元组），并将它们的对应位置的元素 按列 组合成新的元组
a, b, c = zip(*list1) 
# 对于list1，提取所有元组的第1个元素 (1, 3, 5, 7)
# 提取所有元组的第2个元素 (2, 4, 6, 8)
# 提取所有元组的第3个元素 ('a', 'b', 'c', 'd')
# 再解包操作
print(a) # (1, 3, 5, 7)
print(b) # (2, 4, 6, 8)
print(c) # ('a', 'b', 'c', 'd')

# =============================================================================
# basic 3
# =============================================================================

# -------------------- 获取字典的键与值 ---------------------
d = {"a" : 0, "b" : 1, "c" : 2}
# 获取值
v = tuple(d.values())
# 获取键
k = list(d.keys())

print(type(d.values())) # <class 'dict_values'>
print(type(d.keys())) # <class 'dict_keys'>

# -------------------- 字典的添加、删除与修改 ---------------------
# 修改
d["a"] = "c"
print(d) # {'a': 'c', 'b': 1, 'c': 2}
d.update({"a" : "d"})
print(d) # {'a': 'd', 'b': 1, 'c': 2}

# 添加
my_dict = {}
my_dict['key'] = 'value'
print(my_dict) # {'key': 'value'}

# update()添加元素
my_dict.update({'key1' : 'value1'})
print(my_dict) # {'key': 'value', 'key1': 'value1'}

# setdefault() 添加单个元素，如果键已存在，则返回现有的值，否则将键和默认值添加到字典
my_dict = {'key' : '1'} 
my_dict.setdefault('key', 'value')
print(my_dict) # {'key': '1'}
my_dict.setdefault('value', '2')
print(my_dict) # {'key': '1', 'value': '2'}

# 删除
d = {"a" : 0, "b" : 1, "c" : 2}
d.pop("a")
del d["b"]
print(d) # {'c': 2}

d = {"a" : 0, "b" : 1, "c" : 2}
d.popitem() # 删除最后一个插入的键值对
print(d) # {'a': 0, 'b': 1}

# 清空整个字典
d.clear()
print(d) # {}

# 获取某个键的值
d = {"a" : 0, "b" : 1, "c" : 2}
print(d["a"])
# 如果输出不存在的键的值，索引会报错，但使用get不会报错
# print(d["f"]) # KeyError: 'f'
print(d.get("f", "Sorry, can not find it.")) # Sorry, can not find it.
print(d.get("f")) # None

# -------------------- 字典与列表的嵌套 ---------------------
list1 = [ {"学号" : 1234, "姓名" : "布丁", "sex" : "nan"}, 
          {"学号" : 123, "姓名" : "张三", "sex" : "nan"}, 
          {"学号" : 12345, "姓名" : "李四" , "sex" : "nv"} ]

nums, names, sexes = zip(*[i.values() for i in list1])

print(nums) # (1234, 123, 12345)
print(names) # ('布丁', '张三', '李四')
print(sexes) # ('nan', 'nan', 'nv')

# -------------------- 字典不支持切片 ---------------------
dict1 = {"a" : 0, "b" : 1, "c" : 2}
# print(dict1[:]) # TypeError: unhashable type: 'slice'

# =============================================================================
# basic 4
# =============================================================================

# -------------------- set 集合 ---------------------

# 集合的特点是不重复、无序，因为无序所以不支持下标访问
set1 = { 1, 2, 3 }
# print(set1[0]) # TypeError: 'set' object is not subscriptable

# -------------------- 给列表去重 ---------------------
# 使用set可以给list去重
list1 = [ 1, 2, 1, 3, 4]
list2 = list(set(list1))
print(list2) # [1, 2, 3, 4]

# -------------------- 集合遍历 ---------------------
for i in set1:
    print(i)

# 因为不能使用下标访问集合，所以下面的方式是错误的
# for i in range(len(set1)):
#     print(set1[i]) # TypeError: 'set' object is not subscriptable

# =============================================================================
# basic 5
# =============================================================================

# bool类型
a = True
b = False

# if 遇到 0, None, False, [], "", {} 会认为是空，即假
if "2":
    print("Hello")
    
if b:
    print("world")

# set在结合if使用的时候会去掉重复值
set1 = { 1, 2, 3, 1}
set2 = { 1, 2, 2, 3}
if set1 == set2:
    print("Yes")

# 使用id获取变量地址
a = 30
b = 30
if id(a) == id(b):
    print("Yes")
    
# 整数池
a = 300000000000000000000000000000000
b = 300000000000000000000000000000000

print(id(a))
print(id(b))

c = "123"
d = "123"
print(id(c))
print(id(d))

# =============================================================================
# basic 6
# =============================================================================

a = range(0, 10, 1)
for i in a:
    print(i)
    
a = [1, 2, 3, 4]
b = [2, 3, 4, 5]
print(a + b)