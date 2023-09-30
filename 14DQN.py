"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0', render_mode="human")  # Added render_mode
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

# 神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)#初始化输入层，随机生成最开始参数的值
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

# Deep Q learning 框架
class DQN(object):
    def __init__(self):
        # 将eval——net的参数转换到框架给的参数
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0# 学习步数
        # 初始化记忆库
        self.memory_counter = 0 
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))# 存的行数，与多少
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    # 对环境做出行为
    def choose_action(self, x):
        if len(x) != 4:  # CartPole state should have 4 elements
            raise ValueError("Invalid input for choosing action")
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # input only one sample
        if np.random.uniform() < EPSILON:  # greedy，选取最高值的动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)   
        return action

    # 记忆库
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))# 将记忆打包
        index = self.memory_counter % MEMORY_CAPACITY   # 超过记忆上线就重新开始索引
        self.memory[index, :] = transition 
        self.memory_counter += 1
    # 学习过程，从记忆库中提取记忆然后进行
    def learn(self):
        # 检测要不要更新，隔多少步更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())# 将evalnet参数更新到targetnet
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :] # 从记忆库中随机抽取记忆
        # 打包记忆
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # 根据当初施加动作上动作的价值与目标算出loss
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(400):
    s_tuple = env.reset()  # Get the tuple containing the state and an empty dict
    s = np.array(s_tuple[0])  # Extract only the array part of the state
    print(f"Initial state s: {s}")  # Debug print
    ep_r = 0
    while True:
        env.render()
        a = dqn.choose_action(s)  # Pass only the array part of the state
        # Proper unpacking of return values from env.step()
        # s_, r, done, info = env.step(a)  
        s_, r, done, _, info = env.step(a)  # The underscore absorbs the extra value
        x, x_dot, theta, theta_dot = s_
        # 修改的reward
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        # 存储现在的反馈，之前状态，施加动作，环境奖励
        dqn.store_transition(s, a, r, s_)
        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep:', i_episode,
                      '| Ep_r:', round(ep_r, 2))
        if done:
            break
        s = s_