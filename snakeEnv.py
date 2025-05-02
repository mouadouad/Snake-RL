import numpy as np
import gymnasium
from gymnasium import spaces
import matplotlib.pyplot as plt

from SnakeGame.snakeGame import SnakeGame

class Snake(gymnasium.Env):
    def __init__(self, pixels, obs_size, player2_model):
        super(Snake, self).__init__()
        self.game = SnakeGame(pixels, obs_size, player2_model)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict({
            # 'board': spaces.Box(low=-2, high=2, shape=self.game.observation_spec(), dtype=np.int32),
            'local': spaces.Box(low=-2, high=2, shape=(1, obs_size, obs_size), dtype=np.int32),
            'distance': spaces.Box(low=0, high=40, shape=(3,), dtype=np.int32),
            'obstacle_ahead': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),

        }) 
        # self.observation_space = spaces.Box(low=0, high=2, shape=self.game.observation_spec(), dtype=np.float32)
        self.is_render = False
        self.cax = None

    def reset(self, **kwargs):
        return self.game.reset()

    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        if self.is_render:
            self.cax.set_data(self.game.board.board)
            plt.draw()
            plt.pause(0.1)
        return obs, reward, done, False, info

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

if __name__ == "__main__":
    env = Snake(pixels=40, obs_size=11, player2_model=None)
    from stable_baselines3.common.env_checker import check_env

    check_env(env)
    