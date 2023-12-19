from SnakeGame.snake import Snake
from SnakeGame.directions import Directions
from SnakeGame.board import Board

from abc import ABC
import numpy as np
import random
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts


class SnakeGame(py_environment.PyEnvironment, ABC):
    def __init__(self, pixels):
        super().__init__()
        self.pixels = pixels
        self.board = Board(pixels)
        side = random.randint(0, 4)
        self.my_snake = Snake(side, self.board.rows_count, self.board.columns_count)
        self.board.starting_position(self.my_snake.head)
        self.isPlaying = True
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
        self.board = Board(self.pixels)
        side = random.randint(0, 4)
        self.my_snake = Snake(side, self.board.rows_count, self.board.columns_count)
        self.board.starting_position(self.my_snake.head)
        self.isPlaying = True
        self.score = 0

        return ts.restart(self.board.observation(self.my_snake.head))

    def _step(self, action):
        self.score += 1
        if not self.isPlaying:
            return self.reset()

        direction = self.my_snake.direction
        head = self.my_snake.head

        next_direction = direction  # go straight (default)

        if action == 1:   # turn right
            next_direction = list(Directions)[(direction.value + 1) % 4]
        elif action == 2:  # turn left
            next_direction = list(Directions)[(direction.value - 1) % 4]

        next_position = Board.next_position(head, next_direction)

        if self.board.can_advance(*next_position):
            self.board.advance(head, next_position)
            self.my_snake.set_head(next_position)
            self.my_snake.set_direction(next_direction)
            reward = 1.02**self.score
            return ts.transition(self.board.observation(next_position), reward)
        else:
            self.isPlaying = False
            reward = -10
            return ts.termination(self.board.observation(next_position), reward)
