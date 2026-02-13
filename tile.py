from enum import Enum


class Tile(Enum):
    WALL = ("W", (0, 0, 255))
    HEAD = ("H", (0, 120, 255))
    BODY = ("B", (0, 0, 200))
    GREEN = ("G", (0, 255, 0))
    RED = ("R", (255, 0, 0))
    EMPTY = ("0", (200, 200, 200))

    def __init__(self, char, color):
        self.char = char
        self._color = color

    @property
    def color(self):
        return self._color
