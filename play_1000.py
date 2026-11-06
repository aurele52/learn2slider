# play_1000.py
from pathlib import Path
import sys
import math
import random
import time

from environement import Environment
from interpreter import Interpreter
from agent import Agent
from utils import intDir

# IMPORTANT (v6):
# Agent.registre is keyed by packed+canonical INT, not by the raw state tuple.
# So evaluation must canonicalize+pack the state and read Q-values with
# that key.
from agent import _canonical_pack_key, _transform_state, _canon_to_env_action


def greedy_action_no_suicide(agent: Agent, state_env) -> int:
    """
    Returns an ENV action (0..3) using the v6 packed+canonical Q-table.
    Also applies the "no suicide at 1 step" filter in the CANONICAL frame.
    """
    use_mirror = getattr(agent, "use_mirror", True)

    # canonicalize+pack (same as training)
    key, (rot_k, mirror) = _canonical_pack_key(
        state_env, use_mirror=use_mirror
    )
    state_can = _transform_state(state_env, rot_k, mirror)

    # Q-values for canonical key
    values = agent.registre.get(key)

    # actions autorisées = pas mur/body à 1 case (computed in CANONICAL frame)
    allowed_can = []
    for a_can, dirfeat in enumerate(
        state_can
    ):  # (up,right,down,left) in CANONICAL frame
        wall_bin, green_bin, red_bin, body_bin = dirfeat
        if wall_bin == 1 or body_bin == 1:
            continue
        allowed_can.append(a_can)

    if not allowed_can:
        allowed_can = [0, 1, 2, 3]  # obligé

    if values is None:
        a_can = random.choice(allowed_can)
        return _canon_to_env_action(a_can, rot_k, mirror)

    best_v = max(values[a] for a in allowed_can)
    best = [a for a in allowed_can if values[a] == best_v]
    a_can = random.choice(best)  # or best[0] if you want deterministic

    # map canonical action back to ENV action
    a_env = _canon_to_env_action(a_can, rot_k, mirror)
    return a_env


def evaluate(
    model_path: str,
    episodes: int = 1000,
    max_steps_per_ep: int = 10_000,
    seed: int | None = 0,
    render_progress_every: int = 100,
):
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Modèle introuvable: {p}")

    if seed is not None:
        random.seed(seed)

    env = Environment()
    inter = Interpreter(env)

    # Agent en mode "evaluation": pas d'epsilon, pas d'update
    agent = Agent(eps_start=0.0, eps_end=0.0, eps_decay_steps=1, seed=seed)
    agent.load(p)

    print(f"[LOAD] {p} | packed_states={len(agent.registre)}")
    print(f"[MAP CHECK] intDir(0..3) = {[intDir(i) for i in range(4)]}")
    print(f"[AGENT] use_mirror={getattr(agent, 'use_mirror', True)}")

    results = []
    greens = []
    reds = []
    max_lengths = []
    durations = []
    capped = 0
    deaths = 0

    start = time.perf_counter()

    for ep in range(1, episodes + 1):
        env.reset_game()
        ep_green = 0
        ep_red = 0
        max_length = len(env.snake)
        steps = 0
        finished = False

        for _ in range(max_steps_per_ep):
            state = inter.get_state()
            action_env = greedy_action_no_suicide(agent, state)
            direction = intDir(action_env)

            hx, hy = env.snake[0]
            dx, dy = direction
            target = (hx + dx, hy + dy)
            ate_green = target in env.green_apples
            ate_red = target in env.red_apples

            reward, done = inter.apply_dir(direction)
            steps += 1
            max_length = max(max_length, len(env.snake))

            if ate_green:
                ep_green += 1
            if ate_red:
                ep_red += 1

            if done:
                deaths += 1
                finished = True
                break

        if not finished:
            capped += 1

        max_lengths.append(max_length)
        durations.append(steps)

        results.append(len(env.snake))
        greens.append(ep_green)
        reds.append(ep_red)

        if render_progress_every and ep % render_progress_every == 0:
            elapsed = time.perf_counter() - start
            avg_len = (
                sum(results[-render_progress_every:]) / render_progress_every
            )
            avg_g = (
                sum(greens[-render_progress_every:]) / render_progress_every
            )
            avg_r = sum(reds[-render_progress_every:]) / render_progress_every
            print(
                f"ep={ep}/{episodes} elapsed={elapsed:.2f}s "
                f"avgLen(last{render_progress_every})={avg_len:.3f} "
                f"avgGreen={avg_g:.3f} avgRed={avg_r:.3f}"
            )

    # stats finaux
    avg = sum(results) / len(results)
    mn = min(results)
    mx = max(results)
    var = sum((x - avg) ** 2 for x in results) / len(results)
    std = math.sqrt(var)

    avg_g = sum(greens) / len(greens)
    avg_r = sum(reds) / len(reds)

    print("\n=== RESULTATS ===")
    print(f"Parties: {len(results)}")
    print(f"Taille moyenne fin: {avg:.3f}")
    print(f"Min: {mn}")
    print(f"Max: {mx}")
    print(f"Std: {std:.3f}")
    print(f"Moyenne green/partie: {avg_g:.3f}")
    print(f"Moyenne red/partie: {avg_r:.3f}")
    print(f"Deaths: {deaths}")
    print(f"Parties limitees par max_steps: {capped}")
    print(f"Duree moyenne: {sum(durations) / len(durations):.2f}")
    print(f"Duree maximale: {max(durations)}")
    print(f"Longueur maximale atteinte: {max(max_lengths)}")

    success = sum(length >= 10 for length in max_lengths)
    success_final = sum(length >= 10 for length in results)

    print(f"Parties ayant atteint 10: {success}/{episodes}")
    print(f"Taux de reussite: {100 * success / episodes:.2f}%")
    print(f"Parties terminees avec 10+: {success_final}/{episodes}")
    print(f"Pommes vertes mangees: {sum(greens)}")
    print(f"Pommes rouges mangees: {sum(reds)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python play_1000.py train/10000000-v6.pkl [episodes] [max_steps]"
        )
        sys.exit(1)

    model = sys.argv[1]
    episodes = int(sys.argv[2]) if len(sys.argv) >= 3 else 1000
    max_steps = int(sys.argv[3]) if len(sys.argv) >= 4 else 10_000

    evaluate(model, episodes=episodes, max_steps_per_ep=max_steps)
