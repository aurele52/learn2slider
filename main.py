import pygame
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


def intDir(direction):
    # iprint("dirrr", direction)
    if direction == 0:
        return (0, -1)
    if direction == 1:
        return (0, 1)
    if direction == 2:
        return (-1, 0)
    if direction == 3:
        return (1, 0)


def main():
    env = Environment()
    inter = Interpreter(env)
    agent = Agent()

    pygame.init()
    screen_w = env.SQUARE * env.WIDTH + (env.WIDTH + 1) * env.LINE
    screen_h = env.SQUARE * env.HEIGHT + (env.HEIGHT + 1) * env.LINE
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Snake (move on key press)")

    clock = pygame.time.Clock()
    running = True
    for i in range(10000000):
        if i % 100000 == 0:
            print(i // 100000)
        direction = intDir(agent.register(inter.get_state()))
        inter.apply_dir(direction)
        next_state = inter.get_state()
        agent.changeLast(inter.reward, next_state)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # print(agent.getRegistre())
                    inter.reset_game()
                elif event.key in DIRS:
                    direction = intDir(agent.register(inter.get_state()))
                    # print(direction)
                    inter.apply_dir(direction)
                    next_state = inter.get_state()
                    agent.changeLast(inter.reward, next_state)

        env.draw_board(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
