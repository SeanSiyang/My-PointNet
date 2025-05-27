# 1.3.1 算术计算
print("-------------- 1.3.1 -------------------")
print(1 - 2)
print(4 * 5)
print(7 / 5)
print(3 ** 2)

# 1.3.2 数据类型
print("-------------- 1.3.2 -------------------")
print(type(10))
print(type(2.718))
print(type("hello"))

# 1.3.3 变量
print("-------------- 1.3.3 -------------------")
x = 10
print(x)
y = 3.14
print(x * y)
print(type(x * y))

# 1.3.4 列表
print("-------------- 1.3.4 -------------------")
a = [1, 2, 3, 4, 5]
print(a)
print(len(a))

print(a[0])
print(a[4])
a[4] = 99
print(a)

print(a)
print(a[0:2])
print(a[1:])
print(a[:3])
print(a[:-1])
print(a[:-2])

# 1.3.5 字典
print("-------------- 1.3.5 -------------------")
me = { 'height' : 180 }
print(me['height'])
me['weight'] = 70
print(me)

# 1.3.6 布尔型
print("-------------- 1.3.6 -------------------")
hungry = True
sleepy = False
print(type(hungry))
print(not hungry)
print(hungry and sleepy)
print(hungry or sleepy)

# 1.3.7 if语句
print("-------------- 1.3.7 -------------------")
hungry = True
if hungry:
    print("I'm hungry")
    
hungry = False
if hungry:
    print("I'm hungry")
else:
    print("I'm not hungry")
    print("I'm sleepy")

# 1.3.8 for语句
print("-------------- 1.3.8 -------------------")
for i in [1, 2, 3]:
    print(i)


# 1.3.9 函数
print("-------------- 1.3.9 -------------------")
def hello():
    print("Hello World!")
hello()