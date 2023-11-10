import numpy as np
from SnakeGame.directions import Directions


class Board:
    def __init__(self, pixels):
        self.width = 1600
        self.height = 1770
        self.rows_count = int(self.height / pixels)
        self.columns_count = int(self.width / pixels)
        self.board = np.zeros((self.rows_count, self.columns_count), dtype=np.int32)

    def starting_position(self, position):
        self.board[position[0], position[1]] = 2

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

    def advance(self, head, next_position):
        if self.board[head[0], head[1]] != 2:
            raise Exception('Invalid head')

        self.board[head[0], head[1]] = 1
        self.board[next_position[0], next_position[1]] = 2
