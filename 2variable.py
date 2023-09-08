import torch
from torch.autograd import Variable # torch 中 Variable 模块
#神经网络中的参数都是variable变量的形式，引入tensor的数据信息

# 先生鸡蛋
tensor = torch.FloatTensor([[1,2],[3,4]])
# 把鸡蛋放到篮子里, requires_grad是参不参与误差反向传播, 要不要计算梯度
variable = Variable(tensor, requires_grad=True)
#把tensor放入variable，require一般是false

print(tensor)#tensor不能反向传播
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable)#variable可以反向传播
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""
v_out = torch.mean(variable*variable)

#反向传递
v_out.backward()
#v_out=1/4*sum(var*var)
print (variable.grad)#一套体系，所以v_out也有联系