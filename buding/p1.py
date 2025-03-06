# import p2
# print(a2) # NameError: name 'a2' is not defined

# import p2.a2    # ModuleNotFoundError: No module named 'p2.a2'; 'p2' is not a package
# from p2 import a2
# a1 = 10
# print(a2) 

# * 是通配符，代表 all everything
from p2 import *

# 需求：从文件夹里的某个py文件中导入
from import_package.fun1 import * # import_package 与当前文件同级目录


