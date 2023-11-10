from snake import Snake
from directions import Directions
from board import Board


class SnakeGame:
    def __init__(self, pixels):
        self.board = Board(pixels)
        self.my_snake = Snake(0, self.board.rows_count, self.board.columns_count)
        self.board.starting_position(self.my_snake.head)
        self.isPlaying = True

    def run(self, action):
        direction = self.my_snake.direction
        head = self.my_snake.head

        next_direction = direction  # go straight (default)

        if action == 1:   # turn right
            next_direction = list(Directions)[(direction.value + 1) % 4]
        elif action == 2:  # turn left
            next_direction = list(Directions)[(direction.value - 1) % 4]

        next_position = self.board.next_position(head, next_direction)
        if self.board.can_advance(*next_position):
            self.board.advance(head, next_position)
            self.my_snake.set_head(next_position)
            self.my_snake.set_direction(next_direction)
        else:
            self.isPlaying = False
