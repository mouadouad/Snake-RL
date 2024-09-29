from SnakeGame.snake import Snake
from SnakeGame.directions import Directions
from SnakeGame.board import Board

import random
from stable_baselines3 import PPO


class SnakeGame:
    def __init__(self, pixels, obs_size, player2_model=None):
        super().__init__()
        self.pixels = pixels
        self.board = Board(pixels, obs_size)
        if player2_model:
            self.player2_model = PPO.load(f"models/{player2_model}")
        side = random.randint(0, 4)
        self.my_snake = Snake(side, self.board.rows_count, self.board.columns_count)
        self.his_snake = Snake((side + 2) % 4, self.board.rows_count, self.board.columns_count)
        self.board.starting_position(self.my_snake.head, self.his_snake.head)
        self.isPlaying = True
        self.won = False
        self.score = 0

        self._observation_spec = (self.board.obs_size, self.board.obs_size)

    def observation_spec(self):
        return self._observation_spec

    def reset(self):
        self.board.reset()
        side = random.randint(0, 4)
        self.my_snake = Snake(side, self.board.rows_count, self.board.columns_count)
        self.his_snake = Snake((side + 2) % 4, self.board.rows_count, self.board.columns_count)
        self.board.starting_position(self.my_snake.head, self.his_snake.head)
        self.isPlaying = True
        self.won = False
        self.score = 0

        return self.board.observation(self.my_snake.head, self.board.board, self.observation_spec()), dict()

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
            opponent_action = self.player2_model.predict(
                self.board.observation(self.his_snake.head, self.board.board * -1, self.player2_model.observation_space.shape))
            self.opponent_step(opponent_action[0])
        self.score += 1

        next_position, next_direction, head = self.get_next_position(self.my_snake, action)

        if not self.isPlaying:
            reward = 10
            return self.board.observation(next_position, self.board.board, self.observation_spec()), reward, True, dict()

        if self.board.can_advance(*next_position):
            self.board.advance(head, next_position)
            self.my_snake.set_head(next_position)
            self.my_snake.set_direction(next_direction)
            # reward = 1.01**self.score
            reward = 0.2
            return self.board.observation(next_position, self.board.board, self.observation_spec()), reward, False, dict()
        elif not self.won:
            self.isPlaying = False
            reward = -1
            return self.board.observation(next_position, self.board.board, self.observation_spec()), reward, True, dict()
