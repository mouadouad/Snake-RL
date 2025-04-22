import numpy as np
from SnakeGame.directions import Directions


class Board:
    def __init__(self, pixels, obs_size):
        self.width = 1080
        self.height = 1600
        self.obs_size = obs_size
        self.pixels = pixels
        self.rows_count = int(self.height / pixels)
        self.columns_count = int(self.width / pixels)
        self.board = np.zeros((self.rows_count, self.columns_count), dtype=np.int32)

    def reset(self):
        self.board = np.zeros((self.rows_count, self.columns_count), dtype=np.int32)

    def starting_position(self, position, opponent_position):
        self.board[position[0], position[1]] = 2
        self.board[opponent_position[0], opponent_position[1]] = -2

    def can_advance(self, x, y):
        if x < 0 or x >= self.rows_count or y < 0 or y >= self.columns_count:
            return False
        return self.board[x, y] == 0

    @staticmethod
    def next_position(head, direction):
        if direction == Directions.UP:
            return head[0] - 1, head[1]
        elif direction == Directions.RIGHT:
            return head[0], head[1] + 1
        elif direction == Directions.LEFT:
            return head[0], head[1] - 1
        elif direction == Directions.DOWN:
            return head[0] + 1, head[1]

    def advance(self, head, next_position, i=1):
        if self.board[head[0], head[1]] != 2 * i:
            # raise Exception('Invalid head')
            pass

        self.board[head[0], head[1]] = 1 * i
        self.board[next_position[0], next_position[1]] = 2 * i

    def observation(self, head, board, obs_size):
        n = obs_size
        observation = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                x = head[0] - n//2 + i
                y = head[1] - n//2 + j
                if x < 0 or x >= self.rows_count or y < 0 or y >= self.columns_count:
                    observation[i, j] = -1
                else:
                    observation[i, j] = board[x, y]
        return observation.reshape(1, observation.shape[0], observation.shape[1])

    def distance(self, snake):
        straight = 0
        head = tuple(snake.head)
        while self.can_advance(*head):
            straight += 1
            head = self.next_position(head, snake.direction)
        
        right = 0
        head = tuple(snake.head)
        right_direction = list(Directions)[(snake.direction.value + 1) % 4]
        head = self.next_position(head, right_direction)
        while self.can_advance(*head):
            right += 1
            head = self.next_position(head, right_direction)

        left = 0
        head = tuple(snake.head)
        left_direction = list(Directions)[(snake.direction.value - 1) % 4]
        head = self.next_position(head, left_direction)
        while self.can_advance(*head):
            left += 1
            head = self.next_position(head, left_direction)

        return np.array([straight, right, left], dtype=np.int32)