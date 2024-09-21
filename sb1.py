from typing import Callable, Tuple
import os
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import torch.nn as nn
from typing import Tuple
import torch

from SnakeGame.snakeGame import SnakeGame
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

PIXELS = 40
OBS_SIZE = 15
PLAYER2_MODEL = None
MODEL_TO_LOAD = "PPO_17/3980000.zip"


class Snake(gym.Env):
    def __init__(self, pixels, obs_size, player2_model):
        super(Snake, self).__init__()
        self.game = SnakeGame(pixels, obs_size, player2_model)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=2, shape=self.game.observation_spec(), dtype=np.int32)
        self.is_render = False
        self.cax = None

    def reset(self):
        return self.game.reset()

    def step(self, action):
        step = self.game.step(action)
        if self.is_render:
            self.cax.set_data(self.game.board.board)
            plt.draw()
            plt.pause(0.1)
        return *step, {}

    def set_player2_model(self, model_name):
        self.game.set_player2_model(model_name)

    def render(self, mode='human'):
        self.is_render = True
        fig, ax = plt.subplots()
        self.cax = ax.imshow(self.game.board.board, cmap='viridis')
        fig.colorbar(self.cax)
        plt.show(block=False)
        plt.draw()
        plt.pause(0.1)
        
        
        def on_close(event):
            self.is_render = False
            plt.close()
        fig.canvas.mpl_connect('close_event', on_close)
    

    def close(self):
        self.is_render = False
        plt.close()

class CustomNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        output_dim_pi: int = 64,
        output_dim_vf: int = 64,
    ):
        super().__init__()

        self.latent_dim_pi = output_dim_pi
        self.latent_dim_vf = output_dim_vf

        self.num_hidden_layers = num_hidden_layers

        # Policy network
        self.policy_net = self._build_network(feature_dim, hidden_dim, output_dim_pi)

        # Value network
        self.value_net = self._build_network(feature_dim, hidden_dim, output_dim_vf)

    def _build_network(self, input_dim, hidden_dim, output_dim):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(self.num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


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

env = Snake(pixels=PIXELS, obs_size=OBS_SIZE, player2_model=PLAYER2_MODEL)
env.reset()

model = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log=log_dir)

# model = PPO.load(r"models/{MODEL_TO_LOAD}", env, verbose=1, tensorboard_log=log_dir)
for i in range(1, 200):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO_{model_number}")
    model.save(f"{model_dir}/{TIMESTEPS*i}")
    env.set_player2_model(f"PPO_{model_number}/{TIMESTEPS*i}")
    # env.render()
    env.close()
