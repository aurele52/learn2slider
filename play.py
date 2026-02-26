# play.py
import sys
import pygame
from pathlib import Path
from environement import Environment
from interpreter import Interpreter
from agent import Agent

DIRS = {
    pygame.K_UP: (0, -1),
    pygame.K_DOWN: (0, 1),
    pygame.K_LEFT: (-1, 0),
    pygame.K_RIGHT: (1, 0),
}
FPS = 60


def intDir(direction: int):
    if direction == 0:
        return (0, -1)
    if direction == 1:
        return (0, 1)
    if direction == 2:
        return (-1, 0)
    if direction == 3:
        return (1, 0)


def play(load_file: str | None = None, no_random: bool = True):
    env = Environment()
    inter = Interpreter(env)

    # no_random=True => epsilon=0 direct
    agent = Agent(eps_start=0.0 if no_random else 0.2, eps_end=0.0, eps_decay_steps=1)

    if load_file:
        agent.load(Path(load_file))
        print(f"[LOAD] {load_file} (states={len(agent.registre)})")

    pygame.init()
    screen_w = env.SQUARE * env.WIDTH + (env.WIDTH + 1) * env.LINE
    screen_h = env.SQUARE * env.HEIGHT + (env.HEIGHT + 1) * env.LINE
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Snake (1 keypress = 1 step)")

    clock = pygame.time.Clock()
    running = True

    while running:
        stepped = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    inter.reset_game()

                # 1 touche => 1 step (peu importe quelle flèche)
            elif event.key in DIRS:
                direction = intDir(agent.register(inter.get_state()))
                    inter.apply_dir(direction)
                    next_state = inter.get_state()
                    agent.changeLast(inter.reward, next_state)
                    stepped = True

        # on redraw tout le temps (même si pas de step), c'est ok
        env.draw_board(screen)
        pygame.display.flip()

        # si tu veux vraiment figer en attente de touche, tu peux baisser FPS
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    model = None

    # récupère argument: python play.py train/100000-v1.pkl
    if len(sys.argv) > 1:
        model = sys.argv[1]

    play(load_file=model, no_random=True)
