import argparse

from agent import Agent
from environement import Environment
from interpreter import Interpreter
from utils import intDir


def train_sessions(count, save_path, max_steps):
    env = Environment()
    interpreter = Interpreter(env)
    agent = Agent(
        eps_start=0.2,
        eps_end=0.02,
        eps_decay_steps=200000,
        use_mirror=True,
    )

    for episode in range(1, count + 1):
        interpreter.reset_game()
        maximum = len(env.snake)
        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            state = interpreter.get_state()
            action = agent.register(state)
            reward, done = interpreter.apply_dir(intDir(action))
            next_state = interpreter.get_state()

            agent.changeLast(reward, next_state, done)

            total_reward += reward
            maximum = max(maximum, len(env.snake))
            steps += 1

            if done:
                break

        print(
            f"Session {episode}/{count} | "
            f"mouvements={steps} | "
            f"longueur_max={maximum} | "
            f"recompense={total_reward:.2f}",
            flush=True,
        )

    agent.save(save_path)
    print(f"Modele sauvegarde : {save_path}")
    print(f"Etats Q-table : {len(agent.registre)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--max-steps", type=int, default=10000)
    args = parser.parse_args()

    if args.sessions < 1 or args.max_steps < 1:
        parser.error("sessions et max-steps doivent etre positifs")

    train_sessions(args.sessions, args.save, args.max_steps)


if __name__ == "__main__":
    main()
