from typing import Callable
import os
import torch.nn as nn
from typing import Tuple
import torch
import torch as th
from gymnasium import spaces


from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from snakeEnv import Snake

PIXELS = 40
OBS_SIZE = 40
PLAYER2_MODEL = "PPO_47/240000"
MODEL_TO_LOAD = "PPO_47/240000"


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
        layers = list()
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
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        # The features_dim corresponds to the size of the last layer output after CNN
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = 1
        
        # Define the CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: (32, H/2, W/2)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: (64, H/4, W/4)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: (128, H/8, W/8)
            
            nn.Flatten()
        )

        # Compute the size of the flattened output from the CNN
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()  # A sample observation
            print(sample_input.shape)
            n_flatten = self.cnn(sample_input).shape[1]
            print(n_flatten)

        # Create the final fully connected layer that outputs the feature vector
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Pass observations through the CNN and fully connected layer
        return self.linear(self.cnn(observations))

# Step 2: Define a custom ActorCritic policy that uses this CNN feature extractor
class CustomActorCriticCNNPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic Policy with a CNN feature extractor.
    """
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        print(action_space)
        # Specify the feature extractor class (our custom CNN)
        super(CustomActorCriticCNNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),  # Dimensionality of final output of CNN
            **kwargs
        )

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

# net_arch = dict(pi=[256, 128, 128],  
#                  vf=[256, 128, 128])  net_arch=net_arch,

model = PPO(ActorCriticCnnPolicy, env, verbose=1, tensorboard_log=log_dir, device="cuda",
             policy_kwargs=dict( normalize_images=False))

# model = PPO.load(f"models/{MODEL_TO_LOAD}", env, vverbose=1, tensorboard_log=log_dir, device="cuda",
#                   learning_rate=0.0003, ent_coef=0.001, vf_coef=0.5)
for i in range(1, 200):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO_{model_number}")
    model.save(f"{model_dir}/{TIMESTEPS * i}")
    # env.set_player2_model(f"PPO_{model_number}/{TIMESTEPS * i}")
    # env.render()
    # env.close()
