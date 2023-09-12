#@title
import numpy as np

class SnakeEnv():
    def __init__(self):
        self.width = 450
        self.height = 666
        self.left_edge = [0, 0, 20, self.height]
        self.rigth_edge = [self.width-20, 0, self.width, self.height]
        self.top_edge = [0, 0, self.width, 20]
        self.bot_edge = [0, self.height-20, self.width, self.height]
        self.still_traveling = True
        self.my_variables = [[30, self.height-20, self.height-10, 0]]
        self.his_variables = [[0, 0, 0, 0]]
        self.my_counter = 1
        self.his_counter = 1
        self.score = 0

    def preprocessing(self,var):
        # arr = np.zeros((54, 36), dtype=np.float32)
        arr = np.zeros((23, 15), dtype=np.float32)
        last = (0, 0)
        for x in var:
            if x[3] == 0:
                clm = int(x[0] / 30)
                if clm > 35:
                    clm = 35
                for i in range(int(x[1] / 30), int(x[2] / 30)):
                    arr[i][clm] = 1
                    last = (i, clm)
            if x[3] == 180:
                clm = abs(int(x[0] / 30))
                if clm > 36:
                    clm = 36
                for i in range(int(x[1] / 30), int(x[2] / 30)):
                    arr[abs(i) - 1][clm - 1] = 1
                    last = (abs(int(x[1] / 30)) - 1, clm - 1)
            if x[3] == 90:
                row = abs(int(x[0] / 30))
                if x[1] < -self.width:
                    x[1] = -self.width
                for i in range(int(x[1] / 30), int(x[2] / 30)):
                    arr[row][abs(i) - 1] = 1
                    last = (row, abs(int(x[1] / 30)) - 1)
            if x[3] == -90:
                row = abs(int(x[0] / 30))
                for i in range(int(x[1] / 30), int(x[2] / 30)):
                    arr[row - 1][i] = 1
                    last = (row - 1, i)

        arr[last[0], last[1]] = 2
        return arr

    def reset(self):
        self.still_traveling = True
        self.my_variables =[[30, self.height-20, self.height-10, 0]]
        self.his_variables = [[0, 1, 1, 0]]
        self.my_counter = 1
        self.his_counter = 1
        self.bump_self = 0
        self.score = 0
        result = self.preprocessing(self.my_variables)
        # return ts.restart(result)
        return result

    def step(self, action):

        if not self.still_traveling:
            return self.reset()

        # while still_traveling:
        self.score += 1

        if action == 1:
            # turn right
            lastposition = self.my_variables[self.my_counter - 1][3]

            if lastposition == 90:
                lastposition = 180
            elif lastposition == -90:
                lastposition = 0
            elif lastposition == 180:
                lastposition = -90
            elif lastposition == 0:
                lastposition = 90

            first_rect_postition1 = [0, 0, 0, 0]

            first_rect_postition1[0] = self.my_variables[self.my_counter - 1][1]
            first_rect_postition1[1] = -self.my_variables[self.my_counter - 1][0] - 30
            first_rect_postition1[2] = -self.my_variables[self.my_counter - 1][0] - 30
            first_rect_postition1[3] = lastposition

            self.my_variables.append(first_rect_postition1)
            self.my_counter += 1

        elif action == 2:
            # turn left
            lastposition = self.my_variables[self.my_counter - 1][3]

            if lastposition == 90:
                lastposition = 0
            elif lastposition == -90:
                lastposition = 180
            elif lastposition == 180:
                lastposition = 90
            elif lastposition == 0:
                lastposition = -90

            first_rect_postition = [0, 0, 0, 0]

            first_rect_postition[0] = -self.my_variables[self.my_counter - 1][1] - 30
            first_rect_postition[1] = self.my_variables[self.my_counter - 1][0]
            first_rect_postition[2] = self.my_variables[self.my_counter - 1][0]
            first_rect_postition[3] = lastposition

            self.my_variables.append(first_rect_postition)
            self.my_counter += 1

        self.my_variables[self.my_counter - 1][1] -= 30

        # SETTING THE CHECKER
        lastposition = self.my_variables[self.my_counter - 1][3]
        gauche = self.my_variables[self.my_counter - 1][0]
        haut = self.my_variables[self.my_counter - 1][1]

        if lastposition == 90:
            checker = [-haut - 1, gauche, -haut, gauche + 30]
        elif lastposition == -90:
            checker = [haut, -gauche - 30, haut - 1, -gauche]
        elif lastposition == 180:
            checker = [-gauche - 30, -haut - 1, -gauche, -haut]
        else:
            checker = [gauche, haut, gauche + 30, haut + 1]

        for forloop in range(self.my_counter):
            left = self.my_variables[forloop][0]
            top = self.my_variables[forloop][1]
            bot = self.my_variables[forloop][2]
            lastposition = self.my_variables[forloop][3]

            if lastposition == 90:
                rect1 = [-bot, left, -top, left + 30]
            elif lastposition == -90:
                rect1 = [top, -left - 30, bot, -left]
            elif lastposition == 180:
                rect1 = [-left - 30, -bot, -left, -top]
            else:
                rect1 = [left, top, left + 30, bot]

            if checker[0] < rect1[2] and checker[2] > rect1[0] and checker[1] < rect1[3] and checker[3] > \
                    rect1[1] and forloop < self.my_counter - 1:
                if self.bump_self == 0:
                    self.bump_self = 1
                else:
                    self.still_traveling = False

        for forloop in range(self.his_counter):
            left = self.his_variables[forloop][0]
            top = self.his_variables[forloop][1]
            bot = self.his_variables[forloop][2]
            lastposition = self.his_variables[forloop][3]

            if lastposition == 90:
                rect1 = [-bot, left, -top, left + 30]
            elif lastposition == -90:
                rect1 = [top, -left - 30, bot, -left]
            elif lastposition == 180:
                rect1 = [-left - 30, -bot, -left, -top]
            else:
                rect1 = [left, top, left + 30, bot]

            if checker[0] < rect1[2] and checker[2] > rect1[0] and checker[1] < rect1[3] and checker[3] > \
                    rect1[1] and forloop < self.my_counter - 1:
                self.still_traveling = False

        # CHECK IF BUMP TO EDGES
        left_edge = self.left_edge
        right_edge = self.rigth_edge
        top_edge = self.top_edge
        bot_edge = self.bot_edge

        left_b = checker[0] < left_edge[2] and checker[1] < left_edge[3] and \
                 checker[3] > left_edge[1]
        right_b = checker[2] > right_edge[0] and checker[1] < right_edge[3] and \
                  checker[3] > right_edge[1]
        top_b = checker[0] < top_edge[2] and checker[2] > top_edge[0] and checker[1] < top_edge[3]

        bot_b = checker[0] < bot_edge[2] and checker[2] > bot_edge[0] and checker[3] > \
                bot_edge[1]

        if left_b or right_b or top_b or bot_b:
            self.still_traveling = False

        result = self.preprocessing(self.my_variables)

        if self.still_traveling:
            reward = 1  #1.02**self.score
            # return ts.transition(result, reward)
            return result, reward, not (self.still_traveling)
        else:
            reward = 0
            # return ts.termination(result, reward)
            return result, reward, not (self.still_traveling)

import math
import random
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from IPython import display

class DQN(nn.Module):
  def __init__(self,height,width):
    super().__init__()

    self.fc1 = nn.Linear(in_features=height*width, out_features=100)
    self.fc2 = nn.Linear(in_features=100, out_features=200)
    self.fc3 = nn.Linear(in_features=200, out_features=200)
    self.fc4 = nn.Linear(in_features=200, out_features=200)
    self.fc5 = nn.Linear(in_features=200, out_features=100)
    self.out = nn.Linear(in_features=100, out_features=3)

  def forward(self,t):
    t = t.flatten()
    t = F.relu(self.fc1(t))
    t = F.relu(self.fc2(t))
    t = F.relu(self.fc3(t))
    t = F.relu(self.fc4(t))
    t = F.relu(self.fc5(t))
    t = self.out(t)
    return t

Experience = namedtuple(
    'Experience',
    ('state','action','next_state','reward')
)

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)

class Agent():
  def __init__(self, strategy, num_actions, device):
    self.current_step = 0
    self.strategy = strategy
    self.num_actions = num_actions
    self.device = device

  def select_action(self, state, policy_net):
    rate = self.strategy.get_exploration_rate(self.current_step)
    self.current_step += 1

    if rate > random.random():
      action = random.randrange(self.num_actions)
      return torch.tensor([action]).to(self.device)
    else:
      with torch.no_grad():
        return policy_net(state).argmax(dim=0).unsqueeze(0).to(self.device)

class SnakeEnvManager():
    def __init__(self, device):
        self.device = device
        self.env = SnakeEnv()
        self.state = self.env.reset()
        self.done = False

    def reset(self):
        self.state = self.env.reset()

    def num_actions_available(self):
        return 3

    def take_action(self, action):
        self.state, reward, self.done = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)

    def get_state(self):
        return torch.from_numpy(self.state).to(self.device)

    def get_height(self):
        return self.state.shape[0]

    def get_width(self):
        return self.state.shape[1]

class QValues():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  @staticmethod
  def get_current(policy_net,states,actions):
    arr = torch.empty(len(states),3).to(QValues.device)
    for i in range(len(states)):
      arr[i] = policy_net(states[i])
    return arr.gather(dim=1, index=actions.unsqueeze(-1))

  @staticmethod
  def get_next(target_net, next_states):
    arr = torch.empty(len(next_states)).to(QValues.device)

    for i in range(len(next_states)):
      arr[i] = target_net(next_states[i]).max(dim=0)[0]

    return arr.detach()

def extract_tensors(experiences):
  batch = Experience(*zip(*experiences))

#   t1 = torch.cat(batch.state).reshape(-1,54,36)
  t1 = torch.cat(batch.state).reshape(-1,23,15)
  t2 = torch.cat(batch.action)
  t3 = torch.cat(batch.reward)
#   t4 = torch.cat(batch.next_state).reshape(-1,54,36)
  t4 = torch.cat(batch.next_state).reshape(-1,23,15)

  return t1,t2,t3,t4

batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 100
memory_size = 100000
lr = 0.001
num_episodes = 10000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
em = SnakeEnvManager(device)

strategy = EpsilonGreedyStrategy(eps_start,eps_end,eps_decay)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemory(memory_size)

policy_net = DQN(em.get_height(), em.get_width()).to(device)
target_net = DQN(em.get_height(), em.get_width()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_return = []
for episode in range(num_episodes):
  em.reset()
  state = em.get_state()

  for timestep in count():
    action = agent.select_action(state,policy_net)
    reward = em.take_action(action)
    next_state = em.get_state()
    memory.push(Experience(state,action,next_state,reward))
    state = next_state

    if memory.can_provide_sample(batch_size):
      experiences = memory.sample(batch_size)
      states, actions, rewards, next_states = extract_tensors(experiences)

      current_q_values = QValues.get_current(policy_net, states, actions)
      next_q_values = QValues.get_next(target_net, next_states)
      target_q_values = (next_q_values * gamma) + rewards

      loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    if em.done:
      episode_return.append(timestep)
      print("Episode", episode)
      print(timestep)
      display.clear_output(wait=True)
      break

  if episode % target_update == 0:
    target_net.load_state_dict(policy_net.state_dict())

rewards_per_thousand_episodes = np.split(np.array(episode_return),num_episodes/100)
count = 100

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/100)))
    count += 100