import random
from SnakeGame.directions import Directions


class Position:
    def __init__(self, side, rows, columns):
        if side == 0:  # left side
            self.direction = Directions.RIGHT
            self.x = random.randint(0, rows - 1)
            self.y = 0
        elif side == 1:  # top side
            self.direction = Directions.DOWN
            self.x = 0
            self.y = random.randint(0, columns - 1)
        elif side == 2:  # right side
            self.direction = Directions.LEFT
            self.x = random.randint(0, rows - 1)
            self.y = columns - 1
        else:  # bottom side
            self.direction = Directions.UP
            self.x = rows - 1
            self.y = random.randint(0, columns - 1)

    def head(self):
        return self.x, self.y

    def direction(self):
        return self.direction


class Snake:
    def __init__(self, side, rows, columns):
        position = Position(side, rows, columns)
        self.head = position.head()
        self.direction = position.direction

    def set_head(self, head):
        self.head = head

    def set_direction(self, direction):
        self.direction = direction
