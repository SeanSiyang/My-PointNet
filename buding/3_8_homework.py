"""
File        : 3_8_homework.py
Author      : Sean Zhang <siyang.zhang1997@outlook.com>
Date        : 2025-03-03
Description : 
    读取文本，将中文放list1，英文放list2，若空则跳过
    用dict编码中文
    用dict统计每个英文字符出现的次数
    
    如果有空的，统一文本使用strip去除前后多余的空白
    给中文编码唯一id时需要确保当前的字符没有在字典中出现过
    统计字符串中的英文字符时，可以使用get函数，也可以直接使用defaultdict
"""

import re
from collections import defaultdict # 可以在访问不存在的键时自动提供默认值

"""
defaultdict的典型使用场景：计算字符串中每个字符出现的次数
from collections import defaultdict

text = "hello world"
char_count = defaultdict(int)

for char in text:
    char_count[char] += 1

print(dict(char_count))
"""

# 预编译正则表达式
CHINESE_PATTERN = re.compile(r'[\u4e00-\u9fff]')    # 中文字符范围
ENGLISH_PATTERN = re.compile(r'[a-zA-Z]')           # 英文字符范围

def is_chinese(s):
    # return bool(re.search('[\u4e00-\u9fff]', s))
    return bool(CHINESE_PATTERN.search(s))

def is_english(s):
    # return bool(re.search('[a-zA-Z]', s))
    return bool(ENGLISH_PATTERN.search(s))

english_list = []
chinese_list = []
english_dict = {}
english_dict_new = defaultdict(int)  # 默认值都为 0

# 为每个中文字符分配唯一ID
chinese_dict = {}
current_id = 0

# 读取文本
data_path = "E:/Codes/手写AI/3_8_基础_匿名函数_sorted/data/text.txt"
with open(data_path, encoding="utf-8") as f:
    book = f.read()
    for text in book:
        text = text.strip()     # 移除首尾空白
        if (is_chinese(text)):
            chinese_list.append(text)
            
            # 需要判断是否处理过，因为要为其分配唯一的id
            if text not in chinese_dict:
                chinese_dict[text] = current_id
                current_id += 1
                
        elif (is_english(text)):
            english_list.append(text)
            # get 在建存在时返回对应的值，不存在的时候返回默认值
            # dict.get(key, default)
            # 避免了手动检查键是否存在
            # english_dict[text] = english_dict.get(text, 0) + 1 
            english_dict_new[text] += 1


# print(english_list)
# print(chinese_list)

# print(english_dict)
print("中文编码字典:", chinese_dict)
print("英文字符计数:", english_dict_new)

# if __name__ == '__main__':
#     pass


import os

def read_data():
    pass

