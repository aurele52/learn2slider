"""Point d’entrée de Learn2Slither."""

import argparse
import sys

import pygame

from game import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-agent", choices=("qtable", "dqn"), default="qtable"
    )
    parser.add_argument("-sessions", type=int, default=1)
    parser.add_argument("-width", type=int, default=10)
    parser.add_argument("-height", type=int, default=10)
    parser.add_argument("-load")
    parser.add_argument("-save")
    parser.add_argument("-visual", choices=("on", "off"), default="on")
    parser.add_argument("-dontlearn", action="store_true")
    parser.add_argument("-step-by-step", action="store_true")
    parser.add_argument("-fps", type=int, default=10)
    parser.add_argument("-max-steps", type=int, default=10000)
    parser.add_argument("-seed", type=int)
    parser.add_argument("-lobby", action="store_true")
    parser.add_argument("-checkpoint-every", type=int, default=1000)
    args = parser.parse_args()

    if args.width < 3 or args.height < 3:
        parser.error("Le plateau doit mesurer au minimum 3x3")

    if args.sessions < 1 or args.max_steps < 1 or args.fps < 1:
        parser.error("sessions, max-steps et fps doivent etre positifs")

    if args.step_by_step and args.visual == "off":
        parser.error("Le mode pas a pas exige -visual on")

    if args.lobby or len(sys.argv) == 1:
        from lobby import show_lobby

        pygame.display.init()
        pygame.font.init()

        configured = show_lobby(args)

        pygame.display.quit()

        if configured is None:
            pygame.quit()
            return

        args = configured

    run(args)


if __name__ == "__main__":
    main()
