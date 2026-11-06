import argparse
import sys
import sys
import sys
import sys
import random
from pathlib import Path

import pygame

from agent import Agent
from environement import Environment
from interpreter import Interpreter
from policy import greedy_action_env
from trainer import save_checkpoint, save_final
from utils import intDir
from hud import HUD_WIDTH, draw_hud, get_q_values

NAMES = ("UP", "RIGHT", "DOWN", "LEFT")
DIRECTIONS = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


def visible_ray(env, dx, dy):
    """Vision brute, utilisée uniquement pour l'affichage."""
    x, y = env.snake[0]
    cells = []

    while True:
        x += dx
        y += dy
        pos = (x, y)

        if not env.in_bounds(x, y) or pos in env.walls:
            cells.append("W")
            break
        if pos in env.snake_set:
            cells.append("S")
        elif pos in env.green_apples:
            cells.append("G")
        elif pos in env.red_apples:
            cells.append("R")
        else:
            cells.append("0")

    return "".join(cells)


def print_vision(env, action, reward, done):
    print("\n=== VISION DEPUIS LA TETE ===")
    for name, (dx, dy) in zip(NAMES, DIRECTIONS):
        print(f"{name:5}: H {visible_ray(env, dx, dy)}")
    print(f"ACTION: {NAMES[action]}")
    print(f"REWARD: {reward:.2f}")
    print(f"GAME OVER: {done}", flush=True)


def wait_for_action(clock, step_mode):
    """Retourne False si la fenêtre est fermée."""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, step_mode

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False, step_mode
                if event.key == pygame.K_SPACE:
                    step_mode = not step_mode
                if event.key == pygame.K_n and step_mode:
                    return True, step_mode

        if not step_mode:
            return True, step_mode

        clock.tick(30)


def run(args):
    if args.seed is not None:
        random.seed(args.seed)

    env = Environment(
        width=args.width,
        height=args.height,
    )
    interpreter = Interpreter(env)

    if args.agent == "dqn":
        from dqn_agent import DQNAgent

        agent = DQNAgent(seed=args.seed)
    elif args.dontlearn:
        agent = Agent(
            eps_start=0.0,
            eps_end=0.0,
            eps_decay_steps=1,
        )
    else:
        agent = Agent(
            eps_start=0.2,
            eps_end=0.02,
            eps_decay_steps=200000,
        )

    if args.load:
        agent.load(args.load)
        print(f"Modele charge : {args.load}")

    visual = args.visual == "on"
    screen = None
    clock = None

    if visual:
        pygame.display.init()
        pygame.font.init()
        width = env.WIDTH * (env.SQUARE + env.LINE)
        height = env.HEIGHT * (env.SQUARE + env.LINE)
        width += env.LINE
        height += env.LINE
        height = max(height, 460)
        screen = pygame.display.set_mode((width + HUD_WIDTH, height))
        pygame.display.set_caption("Learn2Slither")
        clock = pygame.time.Clock()

    cancelled = False
    successes = 0
    total_steps = 0
    max_length_all = 0

    for episode in range(1, args.sessions + 1):
        interpreter.reset_game()
        maximum = len(env.snake)
        steps = 0
        done = False
        step_mode = args.step_by_step
        last_action = None
        last_reward = None
        last_q = None

        print(f"\n===== SESSION {episode}/{args.sessions} =====")

        for _ in range(args.max_steps):
            if visual:
                env.draw_board(screen)
                draw_hud(
                    screen, env, args.agent,
                    episode, args.sessions,
                    steps, maximum,
                    last_action, last_reward,
                    last_q, done,
                )
                pygame.display.flip()
                clock.tick(args.fps)

                proceed, step_mode = wait_for_action(clock, step_mode)
                if not proceed:
                    cancelled = True
                    break

            state = interpreter.get_state()

            if args.dontlearn:
                if args.agent == "dqn":
                    action = agent.choose(state)
                else:
                    action = greedy_action_env(agent, state)
            else:
                action = agent.register(state)

            if visual:
                rays = [visible_ray(env, dx, dy) for dx, dy in DIRECTIONS]
                last_q = get_q_values(agent, state, args.agent)

            reward, done = interpreter.apply_dir(intDir(action))
            next_state = interpreter.get_state()
            last_action = action
            last_reward = reward

            if not args.dontlearn:
                agent.changeLast(reward, next_state, done)

            if visual:
                print("\n=== VISION DEPUIS LA TETE ===")
                for name, ray in zip(NAMES, rays):
                    print(f"{name:5}: H {ray}")
                print(f"ACTION: {NAMES[action]}")
                print(f"REWARD: {reward:.2f}")
                print(f"GAME OVER: {done}", flush=True)

            steps += 1
            maximum = max(maximum, len(env.snake))

            if done:
                break

        if cancelled:
            break

        if visual:
            env.draw_board(screen)
            draw_hud(
                screen, env, args.agent,
                episode, args.sessions,
                steps, maximum,
                last_action, last_reward,
                last_q, done,
            )
            pygame.display.flip()

        total_steps += steps
        max_length_all = max(max_length_all, maximum)
        if maximum >= 10:
            successes += 1

        print(
            f"Session {episode}: mouvements={steps}, "
            f"longueur_finale={len(env.snake)}, "
            f"longueur_max={maximum}, mort={done}"
        )

        save_checkpoint(agent, args, episode)

    if visual:
        pygame.quit()

    save_final(agent, args)

    print("\n=== BILAN ===")
    print(f"Mouvements totaux : {total_steps}")
    print(f"Longueur maximale : {max_length_all}")
    print(f"Sessions avec longueur >= 10 : {successes}")
    if args.agent == "dqn":
        print(f"Transitions en memoire : {len(agent.memory)}")
        print(f"Mouvements appris : {agent.step_count}")
        print(f"Perte DQN : {agent.last_loss}")
    else:
        print(f"Nombre d'etats Q-table : {len(agent.registre)}")
