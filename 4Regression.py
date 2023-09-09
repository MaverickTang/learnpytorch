"""
Dependencies:
torch: 0.4
matplotlib
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible
#fake data linspace把线段分成一点点的
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
#unsqueeze把一维数据变成二维数据
y = x.pow(2) + 0.2*torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
#torch.rand加噪音

# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

## 搭建神经网络
class Net(torch.nn.Module): # 继承torch的模块
    #intialize搭建层所需信息
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() #继承net到模块，官方步骤
        # hidden layer
        # 包含有多少个输入n_feature，隐藏层神经元节点数n_hidden，多少个输出
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  
         # output layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   
    #前向传递的过程
    def forward(self, x):
         # activation function for hidden layer
         # 用激励函数激活隐藏层输出的信息
        x = F.relu(self.hidden(x))     
        #预测时不用激励函数，因为分布会受影响
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network
print(net)  # net architecture

## 优化神经网络
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()   # something about plotting

for t in range(200):
    # 输入x，获得预测值
    prediction = net(x)     # input x and predict based on x
    # 比较预测值与真实值的误差，顺序不要变
    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
    #进行优化，三步都是
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())#画图只支持numpy的数据
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()