from tile import Tile
from typing import List, Tuple
from environement import Environment

Pos = Tuple[int, int]


class Interpreter:
    reward = -1

    def __init__(self, env):
        self.env: Environment = env

    def apply_dir(self, direction):
        newTile = self.env.step(direction)
        if newTile == Tile.WALL:
            self.reward = -100
            self.env.reset_game()
        if newTile == Tile.RED:
            self.reward = -10
        if newTile == Tile.GREEN:
            self.reward = 10
        if newTile == Tile.BODY:
            self.reward = -100
            self.env.reset_game()
        if newTile == Tile.EMPTY:
            self.reward = -1
        # print(self.get_state())
        # print(self.reward)

    def reset_game(self):
        self.env.reset_game()

    def get_state(
        self,
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
        """
        Vision en 4 directions depuis la tête, jusqu'au mur (inclus).
        Renvoie (up, right, down, left), chaque direction = tuple de chars.
        Exemple direction: ('0','G','0','W')
        """
        if not self.env.snake:
            # état "vide" si pas de snake (devrait pas arriver en jeu)
            return (("W",), ("W",), ("W",), ("W",))

        head = self.env.snake[0]
        board: List[List[Tile]] = self.env.get_board()
        hx, hy = head

        def ray(dx: int, dy: int) -> Tuple[str, ...]:
            out: List[str] = []
            x, y = hx, hy
            while True:
                x += dx
                y += dy

                # bord = mur
                if not self.env.in_bounds(x, y):
                    out.append("W")
                    break
                out.append(board[y][x].char)

                # on s'arrête quand on "voit" un mur
                if board[y][x].char == "W":
                    break
            return tuple(out)

        up = ray(0, -1)
        right = ray(1, 0)
        down = ray(0, 1)
        left = ray(-1, 0)

        return (up, right, down, left)
