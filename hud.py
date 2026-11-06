"""Interface minimaliste du jeu Learn2Slither."""

from functools import lru_cache

import pygame

HUD_WIDTH = 260

BG = (17, 24, 39)
WHITE = (238, 242, 247)
MUTED = (148, 163, 184)
ACCENT = (45, 212, 191)
BAR = (71, 85, 105)
LINE = (51, 65, 85)
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT")


@lru_cache(maxsize=8)
def font(size):
    return pygame.font.Font(None, size)


def label(screen, message, x, y, size=22, color=WHITE):
    image = font(size).render(str(message), True, color)
    screen.blit(image, (x, y))


def get_q_values(agent, state, agent_type):
    """Lit les Q-values sans modifier l'agent."""
    if agent_type == "dqn":
        import torch

        values = torch.tensor(
            [agent.encode(state)],
            dtype=torch.float32,
            device=agent.device,
        )
        with torch.no_grad():
            result = agent.policy(values)[0]
        return [float(x) for x in result.tolist()]

    from agent import (
        _canon_to_env_action,
        _canonical_pack_key,
    )

    key, (rotation, mirror) = _canonical_pack_key(
        state, use_mirror=agent.use_mirror
    )
    saved = agent.registre.get(key)
    if saved is None:
        return [0.0] * 4

    values = [0.0] * 4
    for canonical_action in range(4):
        environment_action = _canon_to_env_action(
            canonical_action, rotation, mirror
        )
        values[environment_action] = float(
            saved[canonical_action]
        )
    return values


def draw_hud(
    screen,
    env,
    agent_type,
    episode,
    sessions,
    steps,
    maximum,
    last_action,
    last_reward,
    q_values,
    done,
):
    board_width = (
        env.WIDTH * (env.SQUARE + env.LINE) + env.LINE
    )
    x = board_width
    w = HUD_WIDTH

    pygame.draw.rect(
        screen, BG, (x, 0, w, screen.get_height())
    )
    pygame.draw.line(
        screen, LINE, (x, 0), (x, screen.get_height())
    )

    left = x + 18

    label(screen, "SNAKE AI", left, 18, 34)
    label(
        screen,
        f"{agent_type.upper()}  |  {env.WIDTH}x{env.HEIGHT}",
        left, 54, 19, MUTED,
    )

    pygame.draw.line(
        screen, LINE, (left, 84), (x + w - 18, 84)
    )

    label(screen, "LONGUEUR", left, 98, 19, MUTED)
    label(screen, len(env.snake), left, 119, 48, ACCENT)

    label(
        screen, f"Maximum : {maximum}",
        left, 176, 21,
    )
    label(
        screen, f"Mouvements : {steps}",
        left, 202, 21,
    )
    label(
        screen, f"Session : {episode}/{sessions}",
        left, 228, 19, MUTED,
    )

    if last_action is None:
        action = "-"
    else:
        action = ACTIONS[last_action]

    label(
        screen, f"Action : {action}",
        left, 261, 21,
    )

    if last_reward is not None:
        label(
            screen, f"Reward : {last_reward:.2f}",
            left, 285, 19, MUTED,
        )

    label(screen, "VALEURS Q", left, 318, 20, WHITE)

    if q_values is not None:
        low = min(q_values)
        high = max(q_values)
        span = high - low

        for index, value in enumerate(q_values):
            y = 349 + index * 24

            selected = index == last_action
            color = ACCENT if selected else BAR

            label(
                screen, ACTIONS[index],
                left, y - 2, 18,
                ACCENT if selected else MUTED,
            )

            bar_x = left + 70
            bar_width = 92

            pygame.draw.rect(
                screen, LINE,
                (bar_x, y + 3, bar_width, 9),
                border_radius=3,
            )

            fraction = (
                (value - low) / span if span > 0 else 0.5
            )

            fill = max(0, round(bar_width * fraction))
            if fill:
                pygame.draw.rect(
                    screen, color,
                    (bar_x, y + 3, fill, 9),
                    border_radius=3,
                )

            label(
                screen, f"{value:.1f}",
                left + 174, y - 2, 18,
                WHITE if selected else MUTED,
            )

    if done:
        label(
            screen, "GAME OVER",
            left, screen.get_height() - 27,
            21, ACCENT,
        )
