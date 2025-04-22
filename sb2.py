import os
from snakeEnv import Snake
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO

class BoardAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, patch_size=5, d_model=64, num_heads=4):
        """
        :param features_dim: int, the final feature dimension for the policy/value network.
        :param patch_size: int, size of each square patch (e.g., 5x5).
        :param d_model: int, embedding dimension for each token.
        :param num_heads: int, number of attention heads.
        """
        super(BoardAttentionExtractor, self).__init__(observation_space, features_dim)
        self.patch_size = patch_size

        # Use a convolution to "tokenize" the board.
        # Input channels = board channels (should be 1), output channels = d_model.
        # Kernel size and stride equal the patch size yield non-overlapping patches.
        self.token_conv = nn.Conv2d(in_channels=observation_space.shape[0], 
                                    out_channels=d_model, 
                                    kernel_size=patch_size, 
                                    stride=patch_size)
        
        # After this conv, the board becomes a grid of tokens. We'll flatten the spatial grid.
        # Compute number of tokens from observation dimensions.
        H, W = observation_space.shape[1], observation_space.shape[2]
        self.num_tokens_h = H // patch_size
        self.num_tokens_w = W // patch_size
        self.num_tokens = self.num_tokens_h * self.num_tokens_w

        # Multi-head self-attention layer. With batch_first=True, input shape must be [batch, seq_len, d_model].
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        
        # A final fully connected layer to produce the desired feature dimension.
        self.fc = nn.Linear(d_model, features_dim)

    def forward(self, observations):
        # observations: shape [batch, channels, height, width]
        # Tokenize the board into patches.
        tokens = self.token_conv(observations)
        # tokens shape: [batch, d_model, H/patch_size, W/patch_size]
        
        # Flatten spatial dimensions: [batch, num_tokens, d_model]
        batch_size, d_model, H_tokens, W_tokens = tokens.shape
        tokens = tokens.view(batch_size, d_model, -1).permute(0, 2, 1)  # now [batch, num_tokens, d_model]
        
        # Apply multi-head self-attention.
        # For self-attention, use the same tokens for query, key, and value.
        attn_output, _ = self.attention(tokens, tokens, tokens)
        
        # Pool the output tokens, e.g. via mean pooling across tokens.
        pooled = attn_output.mean(dim=1)  # shape: [batch, d_model]
        
        # Project to final feature dimension.
        features = F.relu(self.fc(pooled))
        return features

PIXELS = 40
OBS_SIZE = 40
PLAYER2_MODEL = None
MODEL_TO_LOAD = "PPO_47/240000"
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

policy_kwargs = dict(
        features_extractor_class=BoardAttentionExtractor,
        features_extractor_kwargs=dict(features_dim=512, patch_size=10, d_model=64, num_heads=8)
    )

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_dir, device="cuda")

for i in range(1, 200):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO_{model_number}")
    model.save(f"{model_dir}/{TIMESTEPS * i}")