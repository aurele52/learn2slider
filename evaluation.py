"""Évaluation des modèles Q-table et DQN."""

import argparse
import random

from environement import Environment
from interpreter import Interpreter
from policy import greedy_action_env
from utils import intDir


def load_agent(agent_type, path):
    if agent_type == "dqn":
        from dqn_agent import DQNAgent

        agent = DQNAgent()
    else:
        from agent import Agent

        agent = Agent(eps_start=0.0, eps_end=0.0)

    agent.load(path)
    return agent


def choose_action(agent, agent_type, state):
    if agent_type == "dqn":
        return agent.choose(state)

    return greedy_action_env(agent, state)


def evaluate(agent, agent_type, sessions, width,
             height, max_steps, seed):
    successes = 0
    final_successes = 0
    total_steps = 0
    total_max_length = 0
    best_length = 0
    deaths = 0

    for episode in range(sessions):
        random.seed(seed + episode)

        env = Environment(width=width, height=height)
        inter = Interpreter(env)
        inter.reset_game()

        maximum = len(env.snake)
        steps = 0

        for _ in range(max_steps):
            state = inter.get_state()
            action = choose_action(agent, agent_type, state)

            _, done = inter.apply_dir(intDir(action))
            steps += 1
            maximum = max(maximum, len(env.snake))

            if done:
                deaths += 1
                break

        successes += maximum >= 10
        final_successes += len(env.snake) >= 10
        total_steps += steps
        total_max_length += maximum
        best_length = max(best_length, maximum)

    print(f"Agent : {agent_type}")
    print(f"Plateau : {width}x{height}")
    print(f"Sessions : {sessions}")
    print(f"Parties ayant atteint 10 : {successes}")
    print(f"Parties terminées à 10+ : {final_successes}")
    print(f"Taux atteint 10 : {100 * successes / sessions:.1f}%")
    print(f"Taux final 10+ : {100 * final_successes / sessions:.1f}%")
    print(f"Longueur maximale : {best_length}")
    print(
        "Longueur maximale moyenne : "
        f"{total_max_length / sessions:.2f}"
    )
    print(
        "Mouvements moyens : "
        f"{total_steps / sessions:.2f}"
    )
    print(f"Parties terminées par une mort : {deaths}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-agent", choices=("qtable", "dqn"), required=True
    )
    parser.add_argument("-load", required=True)
    parser.add_argument("-sessions", type=int, default=100)
    parser.add_argument("-width", type=int, default=10)
    parser.add_argument("-height", type=int, default=10)
    parser.add_argument("-max-steps", type=int, default=1000)
    parser.add_argument("-seed", type=int, default=42)

    args = parser.parse_args()

    if min(
        args.sessions,
        args.width,
        args.height,
        args.max_steps,
    ) < 1:
        parser.error("Les valeurs doivent être positives.")

    if args.width < 3 or args.height < 3:
        parser.error("Plateau minimum : 3x3.")

    agent = load_agent(args.agent, args.load)
    evaluate(
        agent,
        args.agent,
        args.sessions,
        args.width,
        args.height,
        args.max_steps,
        args.seed,
    )


if __name__ == "__main__":
    main()
