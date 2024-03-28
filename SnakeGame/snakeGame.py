from SnakeGame.snake import Snake
from SnakeGame.directions import Directions
from SnakeGame.board import Board

from abc import ABC
import numpy as np
import random
from stable_baselines3 import PPO
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts


class SnakeGame(py_environment.PyEnvironment, ABC):
    def __init__(self, pixels, obs_size, model):
        super().__init__()
        self.pixels = pixels
        self.board = Board(pixels, obs_size)
        self.model = PPO.load(f"models/{model}")
        side = random.randint(0, 4)
        self.my_snake = Snake(side, self.board.rows_count, self.board.columns_count)
        self.his_snake = Snake((side + 2) % 4, self.board.rows_count, self.board.columns_count)
        self.board.starting_position(self.my_snake.head, self.his_snake.head)
        self.isPlaying = True
        self.won = False
        self.score = 0

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.board.obs_size, self.board.obs_size), dtype=np.int32, name='observation')
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def _reset(self):
        self.board.reset()
        side = random.randint(0, 4)
        self.my_snake = Snake(side, self.board.rows_count, self.board.columns_count)
        self.his_snake = Snake((side + 2) % 4, self.board.rows_count, self.board.columns_count)
        self.board.starting_position(self.my_snake.head, self.his_snake.head)
        self.isPlaying = True
        self.won = False
        self.score = 0

        return ts.restart(self.board.observation(self.my_snake.head, self.board.board))

    def set_model(self, model):
        self.model = PPO.load(f"models/{model}")

    @staticmethod
    def get_next_position(snake, action):
        direction = snake.direction
        head = snake.head

        next_direction = direction  # go straight (default)

        if action == 1:  # turn right
            next_direction = list(Directions)[(direction.value + 1) % 4]
        elif action == 2:  # turn left
            next_direction = list(Directions)[(direction.value - 1) % 4]

        return Board.next_position(head, next_direction), next_direction, head

    def opponent_step(self, action):
        next_position, next_direction, head = self.get_next_position(self.his_snake, action)

        if self.board.can_advance(*next_position):
            self.board.advance(head, next_position, -1)
            self.his_snake.set_head(next_position)
            self.his_snake.set_direction(next_direction)
        else:
            self.isPlaying = False
            self.won = True

    def _step(self, action):
        opponent_action = self.model.predict(self.board.observation(self.his_snake.head, self.board.board * -1))
        self.opponent_step(opponent_action[0])
        self.score += 1

        next_position, next_direction, head = self.get_next_position(self.my_snake, action)

        if not self.isPlaying:
            reward = 10
            return ts.termination(self.board.observation(next_position, self.board.board), reward)

        if self.board.can_advance(*next_position):
            self.board.advance(head, next_position)
            self.my_snake.set_head(next_position)
            self.my_snake.set_direction(next_direction)
            # reward = 1.01**self.score
            reward = 1
            return ts.transition(self.board.observation(next_position, self.board.board), reward)
        elif not self.won:
            self.isPlaying = False
            reward = -10
            return ts.termination(self.board.observation(next_position, self.board.board), reward)
        else:
            # reward = 1.01 ** self.score + 10
            reward = 10
            return ts.termination(self.board.observation(next_position, self.board.board), reward)
