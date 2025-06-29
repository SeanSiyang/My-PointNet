# 5.4.1 乘法层的实现
print("-------------- 4.2.2 ----------------")

class MulLayer:
    def __init__(self):
        # 用于保存正向传播时的输入值
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        # 保存正向传播的输入值
        self.x = x
        self.y = y
        out = x * y
        
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        
        return dx, dy

apple = 100
apple_num = 2
tax = 1.1


