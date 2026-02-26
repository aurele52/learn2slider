# play_1000.py
import sys
import math
from pathlib import Path

from tile import Tile
from environement import Environment
from interpreter import Interpreter
from agent import Agent


def intDir(direction: int):
    if direction == 0:
        return (0, -1)
    if direction == 1:
        return (0, 1)
    if direction == 2:
        return (-1, 0)
    if direction == 3:
        return (1, 0)


def greedy_action(agent, state):
    """
    Action 100% déterministe
    -> aucune exploration
    -> aucun impact sur l'agent
    """
    if state not in agent.registre:
        return 0  # fallback neutre

    values = agent.registre[state]
    best_i = 0
    best_v = values[0]

    for i in range(1, 4):
        if values[i] > best_v:
            best_v = values[i]
            best_i = i

    return best_i


def evaluate(model_path, episodes=1000):
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Modèle introuvable: {p}")

    env = Environment()
    inter = Interpreter(env)

    agent = Agent(eps_start=0.0, eps_end=0.0)
    agent.load(p)

    print(f"[LOAD] {p} | states={len(agent.registre)}")

    results = []

    for ep in range(episodes):
        print(ep)
        env.reset_game()

        i = 0
        while True and i < 10000:
            i = i + 1
            state = inter.get_state()

            # 👉 ACTION PURE (aucune modif agent)
            action = greedy_action(agent, state)
            direction = intDir(action)

            new_tile = env.step(direction)

            # fin de partie
            if new_tile in (Tile.WALL, Tile.BODY):
                results.append(len(env.snake))
                break

    # stats
    avg = sum(results) / len(results)
    mn = min(results)
    mx = max(results)
    var = sum((x - avg) ** 2 for x in results) / len(results)
    std = math.sqrt(var)

    print("\n=== RESULTATS ===")
    print(f"Parties: {len(results)}")
    print(f"Taille moyenne fin: {avg:.3f}")
    print(f"Min: {mn}")
    print(f"Max: {mx}")
    print(f"Std: {std:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python play_1000.py model.pkl")
        sys.exit(1)

    evaluate(sys.argv[1])
