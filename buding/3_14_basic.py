#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File        : 3_14_basic.py
Author      : Sean Zhang <siyang.zhang1997@outlook.com>
Date        : 2025-03-05
Description : 
    basic_1 
        - 全局变量
        - 值传递
        - 局部变量
        - 作用域
        
    basic_2
        - 导包
        - module 与 package
    
    basic_3
        - 作业
        
    basic_4
        - 主函数
        - 函数参数
    
    basic_5
        - 类
"""

# =============================================================================
# basic 1
# =============================================================================

# -------------------- 全局变量 ---------------------
b = 20  # 全局变量

# 全局变量可以在函数中使用，但不能在函数中直接修改
# def read_data():
#     print(b)  # 可以访问和使用，但是不能修改
    # b += 1     # UnboundLocalError: local variable 'b' referenced before assignment
#     print(b)
    
# read_data()
# print(b)

# 如果想要在函数中修改全局变量，需要在函数中使用 global 声明全局变量
# def read_data():
#     global b
#     print(b)
#     b += 1
#     print(b)

# read_data()
# print(b)        # 21

# 全局变量不能跟函数的参数重名
# 函数内部的作用域规则会导致参数名覆盖全局函数名，从而引发问题
# 函数参数属于本地作用域（Local），在函数内部会优先使用本地变量名
# 如果参数名与全局函数名相同，参数名会覆盖全局函数名，导致在函数内部无法直接调用原全局函数

# def read_data(b):
#     global b    # SyntaxError: name 'b' is parameter and global
#     b += 1
#     print(b)

# -------------------- 值传递与局部变量 ---------------------
# def read_data(b):   # 参数b是函数内的局部变量
#     b += 1      # 修改的是局部变量b
#     print(b)    # 11
    
# b = 10  # 全局变量
# read_data(b)
# print(b)    # 10    全局变量未被修改

# 需要注意：如果传入的参数是一个可变对象，修改其值不会影响全局，但如果是append添加会影响全局
def modify_list(lst):
    lst.append(4)    # 修改列表内容（影响外部变量）
    lst = [5, 6, 7]  # 重新赋值局部变量 lst（不影响外部变量）

my_list = [1, 2, 3]
modify_list(my_list)
print(my_list)       # 输出 [1, 2, 3, 4]

# 参数传递本质：函数参数接收的是外部变量的 值副本（对不可变对象，如整数）
# 修改参数不会影响外部变量，除非显式使用 global 或操作可变对象（如列表）

# 改进方式：返回具体的结果
# def read_data(b):
#     b += 1
#     return b

# b = 10
# b = read_data(b)
# print(b)    # 11

# for 循环的临时变量在循环外部仍然可以访问
# Python的作用域是 LEGB Local --> Enclosing --> Global --> Built-in
# for、if/else、while 代码块不会创建新的作用域
# 在全局作用域（模块层级）中的for循环里，i会成为全局变量
# 在函数内部定义的for循环，i会成为该函数的局部变量

# list1 = [1, 3, 4, 5]
# for i in list1:
#     print("hello ", i)
# print(i)    # 5
# 存在一个问题，可能会覆盖同作用域下的同名变量，需要留意


# a = 10
# if a == 10:
#     b = 20
# else:
#     c = 30
    
# print(b)
# print(c)    # NameError: name 'c' is not defined

list1 = [13, 3, 4, 5]
b = 0
for i in list1:
    if i == "123":
        c = 20
    if i % 2 == 0:
        a = 10
    else:
        b += 1
print(a)    # for 不是新的作用域，所以 a 可以被访问
print(b)
# print(c)    # NameError: name 'c' is not defined

"""
LEGB作用域: 
    - Local 局部作用域：在函数或方法内部定义的变量
        - 生命周期：从变量定义开始，到函数执行结束
    - Enclosing 嵌套作用域：在嵌套函数（外层函数）中定义的变量，被内层函数访问
        - 内层函数可以读取外层函数的变量，但默认不能修改，需要使用 nonlocal 关键字
        e.g.
            def outer():
                y = 20
                def inner():
                    nonlocal y
                    y = 30
                    print(y)
                inner()
            outer()
        
    - Global 全局作用域：在模块（文件）顶层定义的变量
        - 生命周期：从模块加载开始，到模块被卸载，修改全局变量需要使用global关键字
    - Built-in 内置作用域：Python内置的变量和函数（如 len, print, Exception）
        - 避免覆盖内置函数名
"""

# =============================================================================
# basic 2
# =============================================================================

# -------------------- 导包 ---------------------
# 在同一级目录的不同py之间调用
# e.g. 在p1.py中想调用p2.py中的变量
# 导入的几种方式
# import xxx
# import xxx as xxx
# from xxx import xxx, xxx
# from xxx import xxx as xxx
# from xxx import *
# from xxx.xxx import xxx as xxx

# python 里给包起了别名以后，原名就没法使用了
"""
p2.py:

a2 = 20
b2 = 20

p1.py:

import p2
# print(a2) # NameError: name 'a2' is not defined 因为没有导入进来

# a2是p2.py里面的变量，并不是模块，如果使用.代表的是后面的是模块
# import p2.a2    # ModuleNotFoundError: No module named 'p2.a2'; 'p2' is not a package
from p2 import a2
a1 = 10
print(a2)

# * 是通配符，代表 all everything
from p2 import *

# 需求：从文件夹里的某个py文件中导入
from import_package.fun1 import * # import_package 与当前文件同级目录


# 若要从上一级目录中的某个py文件中导入
# 在import_package的fun1.py中导入上一级目录中p2的内容

# 错误方式
from ..p2 import a2 # ImportError: attempted relative import with no known parent package

# 正确方式
import sys
sys.path.append("..")   # 将上一级添加到工作目录中 
# 需要注意vscode的工作目录是 VSCode 打开的文件夹

from p2 import a2

在导包的时候，会执行引入模块中的全局部分代码，但不会执行里面的主函数
p2.py:
from import_package.fun1 import a
a2 = 20
b2 = 40

import_package/fun1.py:
print("hello")
a = 10
b = 20
c = 30

在执行p2的时候，会输出 hello

如果包与包之间相互调用，就会发生套娃

"""

# -------------------- module 与 package ---------------------
"""
目录结构：
- main.py
- packageA
    - moduleA.py
    - __init__.py

import 一个package的时候，会执行这个package中的__init__.py
在__init__.py中可以放三类文件：
    - 包的初始化
        - 包的环境变量、配置logging
    - 管理包的公关接口，包可以被外部访问的内容
        - 如果希望 from package import x  # x是moduleA中的变量
        - 在__init__.py 中使用相对导入 from .moduleA import x
        - 如果希望 from package import *
        - * 代表的内容就是__init__.py中 __all__ 定义的内容
            - from . import moduleA
            - from .moduleA import x
            - __all__ = ['x', 'moduleA']
        - 在主函数中：
            - from package import *
            print(x)
            print(moduleA.x)
    - 包的信息
        - 在__init__.py中：
            - __version__ = '1.0.0'
            - __author__ = 'Sean'
        - 在主函数中：
            from package import *
            print(package.__version__)
            print(package.__author__)
            
module 称为 模块：一个 .py 文件即是一个模块。模块的名称对应文件名（不含 .py 后缀）
作用：
    - 将代码按功能拆分到不同文件，便于复用和管理
    - 通过 import 语句在其他代码中引用模块
package 称为 包：一个包含 __init__.py 文件 的文件夹称为包，包通过目录结构组织多个模块或子包
    比如：文件夹 package/ 是一个包，其中包含__init__.py 和 子包 effects/、formats/
    
__init__.py 的作用：
    标识该文件夹为 Python 包（即使文件内容为空）
    初始化包的逻辑（如导入子模块或定义 __all__ 列表控制导出内容）

package/                    # 包名：package
├── __init__.py             # 标识为包
├── effects/                # 子包：package.effects
│   ├── __init__.py
│   ├── echo.py             # 模块：package.effects.echo
│   └── reverse.py
└── formats/                # 子包：package.formats
    ├── __init__.py
    └── wavread.py          # 模块：package.formats.wavread
    
模块导入：import math_utils
        from math_utils import add
包内模块导入：
        from package.effects import echo
相对导入：在包内部可使用相对路径
        from . import submodule

若包或模块不在Python搜索路径中，需手动添加路径：
    import sys
    sys.path.append("/path/to/your/package")
    
包的层级可无限嵌套，但需确保每层目录均包含 __init__.py 文件

文件夹必须包含 __init__.py 文件才能被视为包
模块名严格对应 .py 文件名，而包名对应文件夹名（无论是否包含其他模块）

模块(Module)：单个 .py 文件，代码功能的最小单元
包(Package)：包含 __init__.py 的文件夹，用于组织模块和子包
命名规范：模块名全小写 + 下划线，包名全小写（避免下划线） 
核心区别：包通过目录结构管理模块，模块通过文件封装代码逻辑

模块的作用：
    - 将相关代码封装到单个文件中，避免命名冲突
    - 通过 import 语句在其他代码中复用模块的功能
    - 支持隐藏实现细节，简化主程序逻辑
包的作用：
    - 通过目录结构管理模块，形成命名空间（如 sound.effects.echo）
    - 通过 __init__.py 初始化包或定义导出的模块列表（__all__ 变量）
    - 支持复杂项目的代码分层，例如框架或库的组织
    
包是模块的容器：包通过目录结构组织模块，形成层次化的命名空间
包本身也是模块：包中的 __init__.py 文件使其成为特殊模块，称为 包模块


"""

# =============================================================================
# basic 3
# =============================================================================

# -------------------- homework ---------------------
"""
作业内容：
读一个文本，该文本每一行前半部分为中文，后半部分为英文，用空格做分隔符

具体需求：
    - 输出两个list, 一个存放中文, 一个存放英文, 如果某一行缺少英文或者中文, 则忽略此行
    - 对中文的每个汉字做映射word to index, key是汉字, value是对应的索引
    - 对英文做一个字典，统计英文字母出现的次数
"""


txt_path = "E:/Codes/手写AI/3_14_导包_函数_作用域_类对象_魔术方法/text.txt"

def read_data(file):
    with open(file, "r", encoding="utf-8") as f:
        # 按行读取
        all_data = f.read().split('\n')
        
        # 其他按行读取方式
        # for line in f:  # 逐行读取，内存高效
        #     print(line.strip())     # 去除行尾换行符
        
        
        
        # 其他按行读取
        # while True:
        #     line = f.readline()
        #     if not line: # 读到文件末尾时退出
        #         break
        #     print(line.rstrip('\n')) # 去除换行符
        
        # 其他读取方式
        # 一次性读取所有行，适合小文件
        # lines = f.readlines()   # 返回列表，每行是列表元素
        # for line in lines:
        #     print(line.strip())
        
    list_c = []
    list_e = []
    
    for data in all_data:
        data_s = data.split(" ") # 将中文和英文分割开  关键代码
        # 判断是否只有一类数据
        if (len(data_s) < 2): # 关键代码
            continue
        
        c, e = data_s # 关键代码
        list_c.append(c)
        list_e.append(e)
        
    return list_c, list_e

# 给每个中文编码
def get_word_2_index():
    c_2_index = {}
    for c in list_c:
        for w in c:
            if w in c_2_index:  # 如果已经处理过了，跳过
                pass
            else:
                # 使用len很巧妙 
                c_2_index[w] = c_2_index.get(w, len(c_2_index)) # 关键代码
    return c_2_index

def get_word_2_index2():
    c_2_index = {}
    n = 0
    for c in list_c:
        for w in c:
            if w in c_2_index:      # 可以通过这种方式判断键是否在字典中
                pass
            else:
                c_2_index[w] = n    # 关键代码
                n += 1
    return c_2_index

# 记录每个字母出现的次数
def get_eng_2_times():
    e_2_times = {}
    for e in list_e:
        for w in e:
            e_2_times[w] = e_2_times.get(w, 0) + 1  # 关键代码
    
    return e_2_times

list_c, list_e = read_data(txt_path)
chinese_2_index = get_word_2_index()
eng_2_times = get_eng_2_times()

print(list_c)
print(list_e)
print(chinese_2_index)
print(eng_2_times)



"""
strip函数
作用：移除字符串 头尾 指定的字符序列（默认为空格、换行符、制表符等空白字符）
仅处理首尾：无法删除中间部分的字符

str.strip([chars])
参数chars: 可选指定要移除的字符集合。若未提供，默认移除空白符

作用：
    - 数据清洗
        user_name = input("用户名：").strip()
    - 文件处理
        with open("data.txt") as f:
            lines = [line.strip() for line in f]

去除所有空格（包括中间和首尾）
s = "h e l l o"
result = s.replace(" ", "")
"""

# in 成员操作符的使用
"""
in 用于快速判断某个元素是否存在于序列（如字符串、列表、元组）、集合或字典的键中。
element in container

若 element 存在于 container 中，返回 True; 否则返回 False
not in 是 in 的反向操作符，用于判断元素 不存在 于容器中

支持的容器类型

序列：字符串、列表、元组。
集合: 集合(set)、字典(dict, 默认检查键)

使用场景和技巧：
- 条件判断：数据过滤：快速筛选符合条件的元素
    users = ["Alice", "Bob", "Charlie"]
    if "Bob" in users:
        print("用户存在")
        
- 循环遍历：与 for 循环结合，迭代容器中的每个元素
    fruits = ["apple", "banana", "cherry"]
    for fruit in fruits:
        print(fruit)
- 数据清洗：清理字符串中的无效字符
    user_input = "  user@example.com  "
    if "@" in user_input.strip():
        print("有效邮箱")
- 性能优化：当需要频繁检查元素是否存在时，集合或字典的 in 操作效率远高于列表或元组
    large_list = [i for i in range(100000)]
    large_set = set(large_list)
    # 列表查找（慢）
    if 99999 in large_list:
        pass
    # 集合查找（快）
    if 99999 in large_set:
        pass
"""


# =============================================================================
# basic 4
# =============================================================================

# -------------------- 主函数 ---------------------

# import os

# def fun1():
#     pass

# if __name__ == "__main__":
#     pass

# from p2 import a2, a3

# if __name__ == "__main__":
#     print(a2)
#     print(a3)

# 需要注意调用的模块里面的主函数的内容是不执行的，如果调用的某个变量是主函数里面的，会报错
# p2.py

# a2 = 20
# b2 = 40
# # a3 = 10

# if __name__ == "__main__":
#     print("hello p2")
    
#     a3 = 200

# fun1.py（执行）
# import os, sys
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parent_dir)

# from p2 import a2, a3

# if __name__ == "__main__":
#     print(a2)
#     # print(a3)   # ImportError: cannot import name 'a3' from 'p2'

# -------------------- 函数参数 ---------------------
# 函数的变参
# 这也是为什么可以给print传无数个参数
# 未命名变参
# def fun2(*a):   # a 接收进来的参数会组成一个 tuple
#     print(type(a))
#     print(a)

# 命名变参
# def fun1(**args):   # args 接收进来的参数会组成一个 字典
#     print(type(args))
#     print(args)
    
# 使用命名变参的时候需要在调用的时候指名参数是谁的
# 在函数声明时，默认参数在右边
# 在函数调用时，默认参数也得在右边
# def fun(a, b, c = 10):
#     print(a, b, c)

# def fun4(a, b, c=10, *d, **e):
#     print(a)
#     print(b)
#     print(c)
#     print(d)
#     print(e)


# if __name__ == "__main__":
#     fun2(1, 2, 3)       # 未命名变参
#     fun1(a = 1, b = 2)  # 命名变参
    
#     # fun(c=1, 2, 3) # SyntaxError: positional argument follows keyword argument
#     fun(a=1, b=2, c=3)
#     # fun(a=1, 3, c=4) # SyntaxError: positional argument follows keyword argument
#     fun(1, b=3, c=4)
    
#     fun4(1, 2, 3, 4, 5, 6, abc=10) 
#     """
#     1
#     2
#     3
#     (4, 5, 6)
#     {'abc': 10}
#     """


# =============================================================================
# basic 5
# =============================================================================

# -------------------- 类 ---------------------

class Apple:
    s = 10  # 类对象 可以通过类别直接访问
    def __init__(self):
        pass
    
a = Apple()
print(a.s)

# 类对象可以通过类别直接访问
class Set_Param:
    lr = 0.001
    epoch = 20
    batch_size = 20

print(Set_Param.lr)
print(Set_Param.epoch)
print(Set_Param.batch_size)

# 类中的函数必须有一个参数，可以叫self，也可以叫 abc 否则会报错
class A:
    a = 10
    def __init__(self):
        print("hello")
    
    def fun1(self):
        print("...")
    
    def fun2(): # TypeError: fun2() takes 0 positional arguments but 1 was given
        print("sss")
    
    def fun3(abc):
        print(abc.a) # 10
        print("sss")

a = A()
a.fun2()   # TypeError: fun2() takes 0 positional arguments but 1 was given

# 使用self在类里调用其他成员函数

# -------------------- 魔术方法 ---------------------
# 在特定情况下会自动触发的方法
# 创建对象时触发 __init__(self, ...)
# print方法会触发 __repr__()
# 释放对象时触发 __del__(self)
# 使用print(obj)或str(obj)时触发 __str__(self)
# __call__(self, ...) 可以让对象像函数一样被调用 obj(args)
# 上下文管理主要使用 __enter__(self), __exit__(self)
# 访问不存在的属性时触发 __getattr__(self, name)
# 设置不存在的属性时触发 __setattr__(self, name, value)
# 访问任何属性时触发 __getattribute__(self, name)
# 使用len(obj)时触发 __len__
# 使用obj[key]时触发 __getitem__
# 使用obj[key] = value时触发 __setitem__
# 配置了__iter__和__next__，可以让对象可迭代，用于for循环
class Counter:
    def __init__(self, max):
        self.max = max
    def __iter__(self):
        self.n = 0
        return self
    def __next__(self):
        if self.n < self.max:
            self.n += 1
            return self.n
        else:
            raise StopIteration


class B:
    def __init__(self, name):     # 初始化对象的时候被调用
        self.name = name
    
    
    
    