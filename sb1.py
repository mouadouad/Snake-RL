import os
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

from SnakeGame.snakeGame import SnakeGame
from stable_baselines3 import PPO


class Snake(gym.Env):
    def __init__(self, pixels, obs_size=15, model_name="PPO_17/920000.zip"):
        super(Snake, self).__init__()
        self.game = SnakeGame(pixels, obs_size, model_name)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=2, shape=self.game.observation_spec().shape, dtype=np.int32)
        self.is_render = False
        self.cax = None

    def reset(self):
        return self.game.reset().observation

    def step(self, action):
        step = self.game.step(action)
        if self.is_render:
            self.cax.set_data(self.game.board.board)
            plt.draw()
            plt.pause(0.1)
        return step.observation, step.reward, step.step_type == 2, {}

    def set_model(self, model_name):
        self.game.set_model(model_name)

    def render(self, mode='human'):
        self.is_render = True
        fig, ax = plt.subplots()
        self.cax = ax.imshow(self.game.board.board, cmap='viridis')
        fig.colorbar(self.cax)
        plt.show(block=False)
        plt.draw()
        plt.pause(0.1)

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

env = Snake(pixels=30, obs_size=15, model_name="PPO_17/920000.zip")
env.reset()

# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

model = PPO.load(r"models/PPO_17/920000.zip", env, verbose=1, tensorboard_log=log_dir)
for i in range(1, 200):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO_{model_number}")
    model.save(f"{model_dir}/{TIMESTEPS*i}")
    env.set_model(f"PPO_{model_number}/{TIMESTEPS*i}")
