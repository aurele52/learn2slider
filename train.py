# train.py
from pathlib import Path
from environement import Environment
from interpreter import Interpreter
from agent import Agent
import time
from utils import intDir

SAVE_DIR = Path("train")
VERSION = "v6"
EXT = "pkl"

SAVE_STEPS = {1_000, 100_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000}


def save_path(step: int) -> Path:
    return SAVE_DIR / f"{step}-{VERSION}.{EXT}"


def train(total_steps: int = 10_000_000):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    env = Environment()
    inter = Interpreter(env)

    # use_mirror=False => rotations only (recommended first)
    # set use_mirror=True to also merge reflections
    agent = Agent(
        eps_start=0.2,
        eps_end=0.02,
        eps_decay_steps=2_000_000,
        use_mirror=True,
    )

    state = inter.get_state()

    # --- stats ---
    log_every = 100_000
    r_sum = 0.0
    deaths = 0
    green = 0
    red = 0
    len_sum = 0

    def fmt(sec: float) -> str:
        sec = int(sec)
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    for i in range(1, total_steps + 1):
        action_int = agent.register(state)
        direction = intDir(action_int)

        reward, done = inter.apply_dir(direction)
        next_state = inter.get_state()  # observe after step (even if done)

        agent.changeLast(reward, next_state, done)

        # --- stats ---
        r_sum += reward
        if reward == 10:
            green += 1
        elif reward == -10:
            red += 1
        if done:
            deaths += 1
        len_sum += len(env.snake)

        # transition / reset
        if done:
            inter.reset_game()
            state = inter.get_state()
        else:
            state = next_state

        # logging
        if i % log_every == 0:
            now = time.perf_counter()
            elapsed = now - start_time
            steps_per_sec = i / elapsed if elapsed > 0 else 0.0

            remaining = total_steps - i
            eta_sec = remaining / steps_per_sec if steps_per_sec > 0 else 0.0

            avg_r = r_sum / log_every
            avg_len = len_sum / log_every

            print(
                f"step={i} eps={agent.epsilon():.4f} "
                f"states={len(agent.registre)} "
                f"avgR={avg_r:.3f} deaths={deaths} green={green} red={red} avgLen={avg_len:.2f} "
                f"| elapsed={fmt(elapsed)} speed={steps_per_sec:.1f} steps/s eta={fmt(eta_sec)}"
            )

            r_sum = 0.0
            deaths = 0
            green = 0
            red = 0
            len_sum = 0

        # save
        if i in SAVE_STEPS:
            p = save_path(i)
            agent.save(p)
            print(f"[SAVE] {p} (states={len(agent.registre)})")

    total_elapsed = time.perf_counter() - start_time
    print(f"[DONE] total_steps={total_steps} total_time={total_elapsed:.2f}s")


if __name__ == "__main__":
    train(total_steps=10_000_000)
