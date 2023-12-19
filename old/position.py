import random


class Position:
    def __init__(self, side):

        if side == 0:  # left side
            self.angle = 90
            self.left = random.randint(30, 1590)
            self.bottom = -10
            self.top = 0
        elif side == 1:  # top side
            self.angle = 180
            self.left = -random.randint(30, 1080)
            self.bottom = -10
            self.top = 0
        elif side == 2:  # right side
            self.angle = -90
            self.left = -random.randint(30, 1590)
            self.bottom = 1070
            self.top = 1080
        else:  # bottom side
            self.angle = 0
            self.left = random.randint(30, 1080)
            self.bottom = 1590
            self.top = 1600

    def return_position(self):
        return [self.left, self.bottom, self.top, self.angle]
