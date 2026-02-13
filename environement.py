import random
from tile import Tile
import pygame
from collections import deque
from typing import List, Optional, Tuple

Pos = Tuple[int, int]


class Environment:
    HEIGHT, WIDTH = 10, 10
    LINE = 2
    SQUARE = 50

    N_GREEN = 2
    N_RED = 1

    SNAKE_START_LEN = 3

    # set de toutes les tileules possibles (constant)

    def __init__(self) -> None:
        self.ALL_TILES = {
            (x, y) for y in range(self.HEIGHT) for x in range(self.WIDTH)
        }  # state
        self.snake = deque()
        self.snake_set = set()
        self.walls = set()
        self.green_apples = set()
        self.red_apples = set()
        self.direction = (1, 0)

        # free tiles = tout ce qui n'est pas occupé (mis à jour partout)
        self.freeTiles = set(self.ALL_TILES)

        self.reset_game()

    # ----------------------------
    # Low-level helpers (évite forbidden partout)
    # ----------------------------
    def _occupy(self, pos: tuple[int, int]) -> None:
        """Marque une case comme occupée (donc plus libre)."""
        self.freeTiles.discard(pos)

    def _free(self, pos: tuple[int, int]) -> None:
        """Marque une case comme libre."""
        self.freeTiles.add(pos)

    def _random_free_tile(self):
        """Pioche une case libre au hasard (ou None si aucune)."""
        if not self.freeTiles:
            return None
        # random.choice sur tuple/list
        return random.choice(tuple(self.freeTiles))

    def in_bounds(self, x, y):
        return 0 <= x < self.WIDTH and 0 <= y < self.HEIGHT

    def get_board(self) -> List[List[Tile]]:
        """
        Renvoie une grille HEIGHT x WIDTH de Tile.
        O(W*H) mais avec des checks O(1) via sets.
        """
        head: Optional[Pos] = self.snake[0] if self.snake else None

        # On démarre tout en EMPTY
        grid = [[Tile.EMPTY for _ in range(self.WIDTH)] for _ in range(self.HEIGHT)]

        # Pose dans un ordre "priorité" (mur > pommes > serpent)
        for x, y in self.walls:
            if self.in_bounds(x, y):
                grid[y][x] = Tile.WALL

        for x, y in self.green_apples:
            if self.in_bounds(x, y) and grid[y][x] == Tile.EMPTY:
                grid[y][x] = Tile.GREEN

        for x, y in self.red_apples:
            if self.in_bounds(x, y) and grid[y][x] == Tile.EMPTY:
                grid[y][x] = Tile.RED

        # Corps (sans écraser un mur/pomme si jamais bug de spawn)
        for x, y in self.snake_set:
            if self.in_bounds(x, y) and grid[y][x] == Tile.EMPTY:
                grid[y][x] = Tile.BODY

        # Head par-dessus BODY (priorité max côté serpent)
        if head is not None:
            hx, hy = head
            if self.in_bounds(hx, hy) and grid[hy][hx] != Tile.WALL:
                grid[hy][hx] = Tile.HEAD

        return grid

    # ----------------------------
    # Draw
    # ----------------------------
    def draw_board(self, screen):
        screen.fill((0, 0, 0))

        head = self.snake[0] if self.snake else None

        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                tile = (x, y)

                if tile in self.walls:
                    color = Tile.WALL.color
                elif tile in self.green_apples:
                    color = Tile.GREEN.color
                elif tile in self.red_apples:
                    color = Tile.RED.color
                elif tile == head:
                    color = Tile.HEAD.color
                elif tile in self.snake_set:
                    color = Tile.BODY.color
                else:
                    color = Tile.EMPTY.color

                px = x * self.SQUARE + x * self.LINE + self.LINE
                py = y * self.SQUARE + y * self.LINE + self.LINE
                pygame.draw.rect(screen, color, (px, py, self.SQUARE, self.SQUARE))

    # ----------------------------
    # Game init
    # ----------------------------
    def _make_border_walls(self):
        """Optionnel: murs autour. Tu peux retourner set() si tu n'en veux pas."""
        walls = set()
        # for x in range(self.WIDTH):
        #     walls.add((x, 0))
        #     walls.add((x, self.HEIGHT - 1))
        # for y in range(self.HEIGHT):
        #     walls.add((0, y))
        #     walls.add((self.WIDTH - 1, y))
        return walls

    def _rebuild_free_tiles(self):
        """Recalcule freeTiles à partir des sets (utile au reset)."""
        occupied = self.walls | self.snake_set | self.green_apples | self.red_apples
        self.freeTiles = set(self.ALL_TILES - occupied)

    def _place_snake_random(self):
        """Place un snake de longueur SNAKE_START_LEN en ligne droite (sans utiliser forbidden)."""
        snake = deque()

        # pour placer la tête, on évite seulement les murs
        free_no_walls = self.ALL_TILES - self.walls
        if not free_no_walls:
            return snake

        head = random.choice(tuple(free_no_walls))
        hx, hy = head

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(directions)

        for dx, dy in directions:
            candidate = [(hx, hy)]
            ok = True
            for k in range(1, self.SNAKE_START_LEN):
                xk, yk = hx - dx * k, hy - dy * k
                if not self.in_bounds(xk, yk) or (xk, yk) in self.walls:
                    ok = False
                    break
                candidate.append((xk, yk))

            if not ok:
                continue
            if len(set(candidate)) != len(candidate):
                continue

            for pos in candidate:
                snake.append(pos)
            return snake

        snake.append((hx, hy))
        return snake

    def _spawn_green(self):
        tile = self._random_free_tile()
        if tile is None:
            return
        self.green_apples.add(tile)
        self._occupy(tile)

    def _spawn_red(self):
        tile = self._random_free_tile()
        if tile is None:
            return
        self.red_apples.add(tile)
        self._occupy(tile)

    def reset_game(self):
        # reset structures
        self.walls = set(self._make_border_walls())
        self.green_apples = set()
        self.red_apples = set()
        self.snake = deque()
        self.snake_set = set()

        # freeTiles = tout, puis on occupe murs
        self.freeTiles = set(self.ALL_TILES)
        for w in self.walls:
            self._occupy(w)

        # place snake
        snake = self._place_snake_random()
        while len(snake) < 2:
            snake = self._place_snake_random()

        self.snake = snake
        self.snake_set = set(self.snake)
        for p in self.snake_set:
            self._occupy(p)

        # spawn initial apples
        for _ in range(self.N_GREEN):
            self._spawn_green()
        for _ in range(self.N_RED):
            self._spawn_red()

        # direction initiale cohérente avec le corps
        hx, hy = self.snake[0]
        nx, ny = self.snake[1]
        init_dir = (hx - nx, hy - ny)
        if init_dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            self.direction = init_dir
        else:
            self.direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])

    # ----------------------------
    # Step (deque + snake_set + freeTiles)
    # ----------------------------

    def step(self, direction):
        """
        Retourne True si vivant, False si mort (collision).
        Règles:
        - green: grandit (+1)
        - red: rétrécit (-1 en plus du mouvement)
        """
        self.direction = direction
        if not self.snake:
            return False

        hx, hy = self.snake[0]
        dx, dy = direction
        nx, ny = hx + dx, hy + dy
        next_pos = (nx, ny)

        # collision bounds / walls
        if not self.in_bounds(nx, ny) or next_pos in self.walls:
            return Tile.WALL

        ate_green = next_pos in self.green_apples
        ate_red = next_pos in self.red_apples

        # Combien de segments de queue vont disparaître CE tour ?
        # - si green: 0 (on grandit, pas de pop)
        # - sinon: 1 (mouvement normal)
        # - si red en plus (et pas green): +1 (rétrécit)
        remove_count = 0 if ate_green else 1
        if ate_red and not ate_green:
            remove_count += 1

        # Positions "autorisées" dans le corps : les dernières cases qui vont être supprimées
        removable = set()
        if remove_count > 0:
            k = min(remove_count, len(self.snake))
            removable = set(list(self.snake)[-k:])

        # collision snake : OK si on va sur une case qui est en train d'être libérée
        if next_pos in self.snake_set and next_pos not in removable:
            return Tile.BODY

        # --- move head in ---
        self.snake.appendleft(next_pos)
        self.snake_set.add(next_pos)
        self._occupy(next_pos)
        newTile = Tile.EMPTY

        # --- handle green / normal move ---
        if ate_green:
            self.green_apples.remove(next_pos)
            self._spawn_green()
            newTile = Tile.GREEN
            # pas de pop => grandit
        else:
            tail_pos = self.snake.pop()
            # IMPORTANT: si on a avancé sur la queue, tail_pos == next_pos => NE PAS retirer du set / NE PAS libérer
            if tail_pos != next_pos:
                self.snake_set.remove(tail_pos)
                self._free(tail_pos)

        # --- handle red ---
        if ate_red:
            self.red_apples.remove(next_pos)
            self._spawn_red()
            newTile = Tile.RED

            # shrink: pop 1 de plus (uniquement si on n'a pas mangé green)
            if not ate_green and self.snake:
                tail2 = self.snake.pop()
                # Même principe: ne pas retirer/libérer si c'est la nouvelle tête
                if tail2 != next_pos:
                    self.snake_set.remove(tail2)
                    self._free(tail2)

            if len(self.snake) == 0:
                return Tile.WALL

        return newTile
