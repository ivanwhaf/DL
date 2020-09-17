import gym
import time
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np

env = gym.make('CartPole-v0')
episodes = 300
capacity = 2000  # replay memory's capacity
Epsilon = 0.9  # ε
GAMMA = 0.9  # γ
batch_size = 64
lr = 0.01
target_update_freq = 100
train_count = 0


class DQN(nn.Module):
    """DQN network, three full connection layers
    """

    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(16, 2)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class ReplayMemory():
    """structure to store intermediate samples
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = np.zeros((capacity, 10))
        self.counter = 0

    def add_memory(self, memory):
        transition = np.hstack(memory)
        # print('transition:', transition)

        # memory number pass the capacity, cover the origin data
        idx = self.counter % self.capacity
        self.memory[idx, :] = transition
        self.counter += 1

    def get_random_samples(self, batch_size):
        sample_idx = np.random.choice(self.capacity, batch_size)
        samples = self.memory[sample_idx, :]
        # print('batch_samples:', samples)
        return samples

    def __len__(self):
        return len(self.memory)


def choose_action(state, eval_net):
    # choose best action determined by current state(environment)
    # take 'ε-greedy' policy
    state = Variable(torch.FloatTensor(state))
    if random.random() < Epsilon:
        output = eval_net(state)
        action = output.max(0)[1].numpy()
    else:
        action = np.random.randint(0, 2)
    return action


def train(eval_net, target_net, replay_memory, batch_size, optimizer):
    # copy eval net parameters to target net according to update freq
    if train_count % target_update_freq == 0:
        print('Copy eval net parameters to target net')
        target_net.load_state_dict(eval_net.state_dict())

    # get batch samples
    samples = replay_memory.get_random_samples(batch_size)

    # convert numpy data to tensor
    batch_state = Variable(torch.FloatTensor(
        samples[:, :4]), requires_grad=True)
    batch_action = Variable(torch.LongTensor(
        samples[:, 4].astype(int)))
    batch_reward = Variable(torch.FloatTensor(
        samples[:, 5]), requires_grad=True)
    batch_state_ = Variable(torch.FloatTensor(
        samples[:, 6:]), requires_grad=True)

    # calculate value of 'Q' and 'target Q', Bellman: Q(s,a)=r+γ*maxQ(s',a')
    q_eval = eval_net(batch_state).gather(
        1, batch_action.unsqueeze(1)).view(batch_size)
    q_target = batch_reward+GAMMA*target_net(batch_state_).max(1)[0].detach()
    # print('q_eval:', q_eval)
    # print('q_target:', q_target)

    # calculate loss and update gradients
    # loss=y-Q(s,a)=[r+γ*maxQ(s',a')-Q(s,a)]²
    loss = F.mse_loss(q_eval, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    eval_net = DQN()
    target_net = DQN()
    replay_memory = ReplayMemory(capacity)
    optimizer = torch.optim.Adam(eval_net.parameters(), lr=lr)

    for episode in range(episodes):
        state = env.reset()
        for t in range(1000):
            env.render()
            action = choose_action(state, eval_net)
            state_, reward, done, info = env.step(action)

            # modify the reward, very important
            x, x_dot, theta, theta_dot = state_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / \
                env.theta_threshold_radians - 0.5
            reward = r1 + r2

            replay_memory.add_memory((state, [action, reward], state_))

            if replay_memory.counter >= capacity:
                train(eval_net, target_net, replay_memory, batch_size, optimizer)
                train_count += 1
            if done:
                print("Episode {} finished after {} timesteps".format(episode+1, t+1))
                break

            state = state_

    env.close()
    torch.save(eval_net, "./model/dqn_cartploe.pth")
