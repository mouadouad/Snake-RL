import os
import numpy as np
import gym
from gym import spaces

from SnakeGame.snakeGame import SnakeGame
from stable_baselines3 import PPO


class Snake(gym.Env):
    def __init__(self, pixels):
        super(Snake, self).__init__()
        self.game = SnakeGame(pixels)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=2, shape=self.game.observation_spec().shape, dtype=np.int32)

    def reset(self):
        return self.game.reset().observation

    def step(self, action):
        step = self.game.step(action)
        return step.observation, step.reward, step.step_type == 2, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass


log_dir = "logs"
TIMESTEPS = 20000

model_number = 1
model_dir = f"models/PPO_{model_number}"
while os.path.exists(model_dir):
    model_number += 1
    model_dir = f"models/PPO_{model_number}"

os.makedirs(model_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = Snake(30)
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
for i in range(1, 200):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO_{model_number}")
    model.save(f"{model_dir}/{TIMESTEPS*i}")
