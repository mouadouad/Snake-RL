import os
from snakeEnv import Snake
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO

class BoardMultiScaleExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(BoardMultiScaleExtractor, self).__init__(observation_space, features_dim)
        board_shape = observation_space['board'].shape 
        local_shape = observation_space['local'].shape  
        distance_shape = observation_space['distance'].shape

        self.board_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 20x13
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1), #24
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)),
            nn.Flatten()
        )
        board_output_dim = 24 * 5 * 5

        self.local_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        local_output_dim = 16 * local_shape[1] * local_shape[2]

        self.distanc = nn.Sequential(
            nn.Linear(distance_shape[0], 32),
            nn.ReLU()
        )
        distance_output_dim = 32

        total_dim = board_output_dim + local_output_dim + distance_output_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        board_feat = self.board_net(obs['board'])
        local_feat = self.local_net(obs['local'])
        distance_feat = self.distanc(obs['distance'])
        combined = torch.cat([board_feat, local_feat, distance_feat], dim=1)
        return self.mlp(combined)

PIXELS = 40
OBS_SIZE = 9
PLAYER2_MODEL = None
MODEL_TO_LOAD = "PPO_47/240000"
log_dir = "logs"
TIMESTEPS = 30000

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

net_arch = dict(pi=[128],  
                 vf=[128])  

policy_kwargs = dict(
        features_extractor_class=BoardMultiScaleExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=net_arch
    )

model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_dir, device="cuda")

for i in range(1, 200):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO_{model_number}")
    model.save(f"{model_dir}/{TIMESTEPS * i}")