# 若要从上一级目录中的某个py文件中导入
# from ..p2 import a2 # ImportError: attempted relative import with no known parent package

# 正确处理方式
# import sys
# sys.path.append("..")
# from p2 import a2

# def f(a):
#     print("1")

# print("hello")
# a = 10
# b = 20
# c = 30

# import os, sys
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parent_dir)

# from p2 import a2, a3

# if __name__ == "__main__":
#     print(a2)
#     # print(a3)   # ImportError: cannot import name 'a3' from 'p2'

import sys
sys.path.append("./buding/")
from p2 import a2
a2 = 20