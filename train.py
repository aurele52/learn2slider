# train.py
from pathlib import Path
from environement import Environment
from interpreter import Interpreter
from agent import Agent

SAVE_DIR = Path("train")
VERSION = "v2"
EXT = "pkl"

SAVE_STEPS = {1_000, 100_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000}


def intDir(direction: int):
    # 0..3 -> (dx, dy)
    if direction == 0:
        return (0, -1)  # up
    if direction == 1:
        return (0, 1)  # down
    if direction == 2:
        return (-1, 0)  # left
    if direction == 3:
        return (1, 0)  # right
    raise ValueError(f"Invalid direction int: {direction}")


def save_path(step: int) -> Path:
    return SAVE_DIR / f"{step}-{VERSION}.{EXT}"


def train(total_steps: int = 10_000_000):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    env = Environment()
    inter = Interpreter(env)

    # epsilon-greedy décroissant (comme ton Agent actuel)
    agent = Agent(eps_start=0.2, eps_end=0.0, eps_decay_steps=200_000)

    # état initial
    state = inter.get_state()

    for i in range(1, total_steps + 1):
        if i % 100_000 == 0:
            print(f"step={i} eps={agent.epsilon():.4f} states={len(agent.registre)}")

        action_int = agent.register(state)
        direction = intDir(action_int)

        # SOLUCE 1: apply_dir renvoie (reward, done)
        reward, done = inter.apply_dir(direction)

        # attention: si done, l'env a été reset dans apply_dir -> next_state = nouvel état initial
        next_state = inter.get_state()

        # SOLUCE 1: update terminal correct
        agent.changeLast(reward, next_state, done)

        # avance l'état courant
        state = next_state

        if i in SAVE_STEPS:
            p = save_path(i)
            agent.save(p)
            print(f"[SAVE] {p} (states={len(agent.registre)})")


if __name__ == "__main__":
    train(total_steps=10_000_000)
