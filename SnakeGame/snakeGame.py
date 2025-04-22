from SnakeGame.snake import Snake
from SnakeGame.directions import Directions
from SnakeGame.board import Board

import random
from stable_baselines3 import PPO
import numpy as np


class SnakeGame:
    def __init__(self, pixels, obs_size, player2_model=None):
        super().__init__()
        self.pixels = pixels
        self.board = Board(pixels, obs_size)
        if player2_model:
            self.player2_model = PPO.load(f"models/{player2_model}")
        side = random.randint(0, 4)
        # side = 3
        self.my_snake = Snake(side, self.board.rows_count, self.board.columns_count)
        self.his_snake = Snake((side + 2) % 4, self.board.rows_count, self.board.columns_count)
        self.board.starting_position(self.my_snake.head, self.his_snake.head)
        self.isPlaying = True
        self.won = False
        self.score = 0

        self._observation_spec = (1, self.board.rows_count+2, self.board.columns_count+2)

    def observation_spec(self):
        return self._observation_spec

    def reset(self):
        self.board.reset()
        side = random.randint(0, 4)
        # side = 3
        self.my_snake = Snake(side, self.board.rows_count, self.board.columns_count)
        self.his_snake = Snake((side + 2) % 4, self.board.rows_count, self.board.columns_count)
        self.board.starting_position(self.my_snake.head, self.his_snake.head)
        self.isPlaying = True
        self.won = False
        self.score = 0

        return {
            'board': self.preProcess(),
            'local': self.board.observation(self.my_snake.head, self.board.board, self.board.obs_size),
            'distance': self.board.distance(self.my_snake),
        }, dict()
        

    def set_player2_model(self, model):
        if hasattr(self, 'player2_model'):
            self.player2_model = PPO.load(f"models/{model}")

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

    def step(self, action):
        if hasattr(self, 'player2_model'):
            pass
            # opponent_action = self.player2_model.predict(
            #   self.preProcess(self.his_snake).reshape((1, 42, 29))) 
            # self.opponent_step(opponent_action[0])
        self.score += 1

        next_position, next_direction, head = self.get_next_position(self.my_snake, action)
        done = False
        reward = 0

        if action != 0:
            next, _, _ = self.get_next_position(self.my_snake, 0)
            if not self.board.can_advance(*next) and self.board.can_advance(*next_position):
                reward += 0.3
        else:
            reward += 0.1
        
        # if not self.isPlaying:
        #     reward += 3
        #     done = True

        if self.board.can_advance(*next_position):
            self.board.advance(head, next_position)
            self.my_snake.set_head(next_position)
            self.my_snake.set_direction(next_direction)
            # reward += 0.2 + 1.01 ** self.score - 1
            reward += 0.25
        elif not self.won:
            self.isPlaying = False
            reward -= 1
            done = True

        return {
            'board': self.preProcess(),
            'local': self.board.observation(self.my_snake.head, self.board.board, self.board.obs_size),
            'distance': self.board.distance(self.my_snake),
        }, reward, done, dict()

    def preProcess(self):
        return np.pad(
            self.board.board, pad_width=1, mode='constant', constant_values=-1).reshape(
                1, self.board.rows_count+2, self.board.columns_count+2)

    # def preProcess2(self, snake):
    #     bordered_matrix = np.pad(self.board.board, pad_width=1, mode='constant', constant_values=-1)

    #     result = np.zeros(bordered_matrix.shape, dtype=np.float32)
    #     for i in range(bordered_matrix.shape[0]):
    #         for j in range(bordered_matrix.shape[1]):
    #             distance = np.abs(i - snake.head[0]) + np.abs(j - snake.head[1])
    #             result[i][j] = np.abs(bordered_matrix[i][j]) / (distance + 1)
    #     return result.reshape(1, result.shape[0], result.shape[1])