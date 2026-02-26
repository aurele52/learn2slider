# play.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional

import pygame

from environement import Environment
from interpreter import Interpreter
from agent import Agent

# IMPORTANT: v6 Agent stores Q-table with packed+canonical int keys.
# So play.py must canonicalize+pack the state before reading Q-values.
from agent import _canonical_pack_key, _transform_state, _canon_to_env_action

from utils import intDir

Action = int  # 0..3 (UP, RIGHT, DOWN, LEFT) in ENV frame


def allowed_actions_no_suicide_from_state(state_urdl) -> List[int]:
    """
    state_urdl = (up, right, down, left)
    DirFeat = (wall_bin, green_bin, red_bin, body_bin)
    Interdit si wall_bin==1 ou body_bin==1, sauf si obligé.
    IMPORTANT: this must be computed in the SAME frame as the Q-values we use.
    """
    allowed = []
    for a, dirfeat in enumerate(state_urdl):
        wall_bin, green_bin, red_bin, body_bin = dirfeat
        if wall_bin == 1 or body_bin == 1:
            continue
        allowed.append(a)
    return allowed if allowed else [0, 1, 2, 3]


def greedy_action_env(
    agent: Agent, state_env, deterministic_tiebreak: bool = False
) -> Action:
    """
    Returns an action in ENV frame (0..3), using the trained Q-table.
    Works with v6 Agent (canonicalization + packed int keys).
    """
    # canonicalize & pack (same as training)
    key, (rot_k, mirror) = _canonical_pack_key(
        state_env, use_mirror=getattr(agent, "use_mirror", False)
    )
    state_can = _transform_state(state_env, rot_k, mirror)

    # fetch Q-values for this canonical key
    # NOTE: agent.registre is a defaultdict, so agent.registre[key] always exists.
    qvals = agent.registre.get(key)
    if qvals is None:
        # unseen state -> default action
        return 0

    # allowed actions must be computed in CANONICAL frame
    allowed_can = allowed_actions_no_suicide_from_state(state_can)

    # greedy in canonical frame
    best_v = max(qvals[a] for a in allowed_can)
    best = [a for a in allowed_can if qvals[a] == best_v]

    a_can = best[0] if deterministic_tiebreak else random.choice(best)

    # map canonical action back to env action (Option A, direct mapping)
    a_env = _canon_to_env_action(a_can, rot_k, mirror)
    return a_env


def run_episode_headless(
    agent: Agent,
    seed: int,
    max_steps: int,
    deterministic_tiebreak: bool,
) -> Tuple[int, List[Action]]:
    """
    Lance 1 partie SANS affichage, renvoie (taille_finale, actions_env).
    Reproductible via seed.
    """
    random.seed(seed)

    env = Environment()
    inter = Interpreter(env)
    env.reset_game()

    actions_env: List[Action] = []

    for _ in range(max_steps):
        state = inter.get_state()
        a_env = greedy_action_env(
            agent, state, deterministic_tiebreak=deterministic_tiebreak
        )
        actions_env.append(a_env)

        direction = intDir(a_env)
        reward, done = inter.apply_dir(direction)
        if done:
            break

    return len(env.snake), actions_env


def replay_episode_pygame(
    agent: Agent,
    seed: int,
    actions: Optional[List[Action]],
    fps: int,
    step_per_frame: int,
    deterministic_tiebreak: bool,
):
    """
    Rejoue en Pygame.
    - Si actions est fourni: rejoue EXACTEMENT cette suite (actions en repère ENV).
    - Sinon: calcule l'action greedy à chaque step (avec seed pour reproduc).
    """
    random.seed(seed)

    env = Environment()
    inter = Interpreter(env)
    env.reset_game()

    pygame.init()
    screen_w = env.SQUARE * env.WIDTH + (env.WIDTH + 1) * env.LINE
    screen_h = env.SQUARE * env.HEIGHT + (env.HEIGHT + 1) * env.LINE
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption(f"Snake Replay (seed={seed})")

    clock = pygame.time.Clock()
    running = True
    paused = False

    step_i = 0
    max_steps = 10_000_000  # sécurité

    font = pygame.font.SysFont(None, 22)

    while running:
        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # reset et recommence le replay
                    random.seed(seed)
                    env.reset_game()
                    step_i = 0
                elif event.key == pygame.K_n:
                    # step unique si paused
                    if paused:
                        step_per_frame = 1

        if not paused:
            for _ in range(step_per_frame):
                if step_i >= max_steps:
                    paused = True
                    break

                # action du replay (ENV frame)
                if actions is not None:
                    if step_i >= len(actions):
                        paused = True
                        break
                    a_env = actions[step_i]
                else:
                    state = inter.get_state()
                    a_env = greedy_action_env(
                        agent, state, deterministic_tiebreak=deterministic_tiebreak
                    )

                direction = intDir(a_env)
                reward, done = inter.apply_dir(direction)
                step_i += 1

                if done:
                    paused = True
                    break

        # draw
        env.draw_board(screen)

        # overlay info
        txt = f"seed={seed} step={step_i} len={len(env.snake)} paused={'YES' if paused else 'NO'}"
        surf = font.render(txt, True, (255, 255, 255))
        screen.blit(surf, (8, 8))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="ex: train/10000000-v6.pkl")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument(
        "--spf", type=int, default=1, help="steps per frame (autoplay speed)"
    )
    ap.add_argument("--max-steps", type=int, default=10_000)

    # Mode 1: chercher une game min=3
    ap.add_argument(
        "--find-min3",
        action="store_true",
        help="cherche un épisode qui finit len==3 puis le rejoue",
    )
    ap.add_argument(
        "--tries",
        type=int,
        default=50_000,
        help="nombre de seeds testés pour find-min3",
    )
    ap.add_argument("--start-seed", type=int, default=0)

    # Mode 2: rejouer une seed précise
    ap.add_argument("--seed", type=int, default=None)

    # Mode tie-break
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="tie-break déterministe (pas de random sur égalités)",
    )

    # Record/replay file
    ap.add_argument("--save-replay", type=str, default="replay_min3.json")
    ap.add_argument("--load-replay", type=str, default=None)

    args = ap.parse_args()

    p = Path(args.model)
    if not p.exists():
        raise FileNotFoundError(f"Modèle introuvable: {p}")

    agent = Agent(eps_start=0.0, eps_end=0.0, eps_decay_steps=1)
    agent.load(p)
    print(f"[LOAD] {p} | packed_states={len(agent.registre)}")
    print(f"[MAP CHECK] intDir(0..3) = {[intDir(i) for i in range(4)]}")
    print(f"[AGENT] use_mirror={getattr(agent, 'use_mirror', False)}")

    # 1) Load replay file
    if args.load_replay:
        rp = Path(args.load_replay)
        data = json.loads(rp.read_text())
        seed = int(data["seed"])
        actions = list(map(int, data.get("actions", [])))
        print(f"[REPLAY LOAD] {rp} seed={seed} actions={len(actions)}")
        replay_episode_pygame(
            agent,
            seed=seed,
            actions=actions,
            fps=args.fps,
            step_per_frame=args.spf,
            deterministic_tiebreak=args.deterministic,
        )
        return

    # 2) Find min=3
    if args.find_min3:
        best_seed = None
        best_actions = None

        for k in range(args.tries):
            seed = args.start_seed + k
            final_len, actions = run_episode_headless(
                agent,
                seed=seed,
                max_steps=args.max_steps,
                deterministic_tiebreak=args.deterministic,
            )
            if final_len == 3:
                best_seed = seed
                best_actions = actions
                print(f"[FOUND MIN=3] seed={seed} steps={len(actions)}")
                break

        if best_seed is None:
            print(f"[NOT FOUND] Aucun len==3 trouvé sur {args.tries} seeds.")
            return

        # save replay (actions are ENV actions 0..3)
        out = Path(args.save_replay)
        out.write_text(
            json.dumps({"seed": best_seed, "actions": best_actions}, indent=2)
        )
        print(f"[REPLAY SAVE] {out}")

        # visualize
        replay_episode_pygame(
            agent,
            seed=best_seed,
            actions=best_actions,
            fps=args.fps,
            step_per_frame=args.spf,
            deterministic_tiebreak=args.deterministic,
        )
        return

    # 3) Replay a specific seed (compute actions live)
    if args.seed is not None:
        replay_episode_pygame(
            agent,
            seed=args.seed,
            actions=None,  # calcule en live (mais reproductible via seed)
            fps=args.fps,
            step_per_frame=args.spf,
            deterministic_tiebreak=args.deterministic,
        )
        return

    # 4) Default: just run a random seed replay
    seed = random.randint(0, 10**9)
    replay_episode_pygame(
        agent,
        seed=seed,
        actions=None,
        fps=args.fps,
        step_per_frame=args.spf,
        deterministic_tiebreak=args.deterministic,
    )


if __name__ == "__main__":
    main()
