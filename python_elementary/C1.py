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

def hello(object):
    print("Hello " + object + "!")
hello("cat")

# 1.4.2 类
print("-------------- 1.4.2 -------------------")
class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized")
    
    def hello(self):
        print("Hello " + self.name + "!")
    
    def goodbye(self):
        print("Good-bye " + self.name + "!")
    
m = Man("David")
m.hello()
m.goodbye()

# 1.5.1 导入Numpy
print("-------------- 1.5.1 -------------------")
import numpy as np

# 1.5.2 生成Numpy数组
print("-------------- 1.5.2 -------------------")
x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))

# 1.5.3 Numpy的算术运算
print("-------------- 1.5.3 -------------------")
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x + y)
print(x - y)
print(x * y)
print(x / y)

x = np.array([1.0, 2.0, 3.0])
print(x / 2.0)

# 1.5.4 Numpy的N维数组
print("-------------- 1.5.4 -------------------")
A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)
print(A.dtype)

B = np.array([[3, 0], [0, 6]])
print(A + B)
print(A * B)    # 逐元素相乘

# 1.5.5 广播
print("-------------- 1.5.5 -------------------")
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print(A * B)

# 1.5.6 访问元素
print("-------------- 1.5.6 -------------------")
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0])
print(X[0][1])

for row in X:
    print(row)

X = X.flatten()     # 将X转换为一维数组
print(X)
print(X[np.array([0, 2, 4])])
print(X > 15)
print(X[X>15])

# 1.6.1 绘制简单图形
print("-------------- 1.6.1 -------------------")
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y = np.sin(x)

# 绘制图形
# plt.plot(x, y)  
# plt.show()

# 1.6.2 pyplot的功能
print("-------------- 1.6.2 -------------------")
# 生成x轴数据
x = np.arange(0, 6, 0.1)
# 生成对应的正弦函数值数组
y1 = np.sin(x)
# 生成对应的余弦函数值数组
y2 = np.cos(x)

# 绘制图形，使用pyplot添加标题和x轴标签名
# 绘制正弦曲线，并设置图例标签为sin
plt.plot(x, y1, label="sin")
# 绘制余弦曲线，并设置图例标签为cos
plt.plot(x, y2, linestyle="--", label="cos")
# 设置x轴标签为x
plt.xlabel("x")             # x轴标签名
# 设置y轴标签为y
plt.ylabel("y")             # y轴标签名
# 设置图标标题
plt.title('sin & cos')      # 添加标题
# 显示图例（根据plot函数中的label参数自动生成）
plt.legend()
# 显示绘制好的图表
plt.show()
