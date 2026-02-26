# utils.py


def intDir(direction: int):
    # 0..3 -> (dx, dy) : UP, RIGHT, DOWN, LEFT
    if direction == 0:
        return (0, -1)  # up
    if direction == 1:
        return (1, 0)  # right
    if direction == 2:
        return (0, 1)  # down
    if direction == 3:
        return (-1, 0)  # left
    raise ValueError(f"Invalid direction int: {direction}")
