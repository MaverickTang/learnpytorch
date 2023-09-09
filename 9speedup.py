import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

def generate_fake_data():
    x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
    y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))
    return x, y

def load_data(x, y, batch_size=32):
    torch_dataset = Data.TensorDataset(x, y)
    return Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def plot_loss(losses_his, labels):
    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)  # hidden layer
        self.predict = torch.nn.Linear(20, 1)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x

if __name__ == '__main__':
    LR = 0.01
    BATCH_SIZE = 32
    EPOCH = 12
    
    x, y = generate_fake_data()
    plt.scatter(x.numpy(), y.numpy())  # plot dataset
    plt.show()
    
    loader = load_data(x, y, BATCH_SIZE)
    
    nets = [Net() for _ in range(4)]
    opt_SGD = torch.optim.SGD(nets[0].parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(nets[1].parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(nets[2].parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(nets[3].parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]
    
    loss_func = torch.nn.MSELoss()
    losses_his = [[], [], [], []]  # record loss

    for epoch in range(EPOCH):
        print(f'Epoch: {epoch}')
        for step, (batch_x, batch_y) in enumerate(loader):
            for net, opt, loss_history in zip(nets, optimizers, losses_his):
                output = net(batch_x)
                loss = loss_func(output, batch_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_history.append(loss.data.numpy())

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    plot_loss(losses_his, labels)
