# interpreter.py
from typing import Tuple

from environement import Environment
from tile import Tile

# Etat compact: (up, right, down, left)
# chaque direction: (wall_bin, green_bin, red_bin, body_bin)
DirFeat = Tuple[int, int, int, int]
StateType = Tuple[DirFeat, DirFeat, DirFeat, DirFeat]


def manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def closest_green_dist(env) -> int | None:
    if not env.green_apples or not env.snake:
        return None
    h = env.snake[0]
    return min(manhattan(h, g) for g in env.green_apples)


class Interpreter:
    def __init__(self, env):
        self.env: Environment = env

    def apply_dir(self, direction):
        # distance avant le move (vers la green la plus proche)

        d0 = closest_green_dist(self.env)

        newTile = self.env.step(direction)

        done = False
        if newTile in (Tile.WALL, Tile.BODY):
            reward = -100
            done = True
        elif newTile == Tile.RED:
            reward = -10
        elif newTile == Tile.GREEN:
            reward = 10
        else:
            reward = -0.01

        # distance après le move
        d1 = closest_green_dist(self.env)

        # reward shaping (seulement si on n'est pas mort et qu'on a des greens)
        if (
            (not done)
            and (d0 is not None)
            and (d1 is not None)
            and (newTile != Tile.GREEN)
        ):
            # réglage: commence petit
            SHAPE = 0.2
            if d1 < d0:
                reward += SHAPE
            elif d1 > d0:
                reward -= SHAPE

        return reward, done

    def reset_game(self):
        self.env.reset_game()

    # ---------- NEW: binning ----------
    @staticmethod
    def _bin_dist(d: int) -> int:
        """
        Bins demandés:
        0 = absent (uniquement pour apple/body)
        1 = 1
        2 = 2
        3 = 3..5
        4 = 6..inf
        """
        if d <= 0:
            return 0
        if d == 1:
            return 1
        if d == 2:
            return 2
        if 3 <= d <= 5:
            return 3
        return 4

    def _ray_features(self, dx: int, dy: int) -> DirFeat:
        """
        Calcule distances jusqu'à:
        - mur (bord ou wall set) [toujours présent]
        - green / red / body (0 si absent avant le mur)
        Puis applique binning.
        """
        if not self.env.snake:
            return (4, 0, 0, 0)

        hx, hy = self.env.snake[0]

        dist = 0
        seen_green = 0
        seen_red = 0
        seen_body = 0

        x, y = hx, hy
        while True:
            dist += 1
            x += dx
            y += dy

            # bord = mur
            if not self.env.in_bounds(x, y) or (x, y) in self.env.walls:
                wall_bin = self._bin_dist(dist)
                return (
                    wall_bin,
                    self._bin_dist(seen_green),
                    self._bin_dist(seen_red),
                    self._bin_dist(seen_body),
                )

            pos = (x, y)
            # On ne garde que la 1ère occurrence (distance minimale)
            if seen_green == 0 and pos in self.env.green_apples:
                seen_green = dist
            if seen_red == 0 and pos in self.env.red_apples:
                seen_red = dist
            if seen_body == 0 and pos in self.env.snake_set:
                seen_body = dist

    def get_state(self) -> StateType:
        # ordre: up, right, down, left (comme tu avais)
        up = self._ray_features(0, -1)
        right = self._ray_features(1, 0)
        down = self._ray_features(0, 1)
        left = self._ray_features(-1, 0)
        return (up, right, down, left)
