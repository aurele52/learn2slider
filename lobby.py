"""Menu d'accueil et configuration de Learn2Slither."""

from datetime import datetime
from pathlib import Path

import pygame


WIDTH, HEIGHT = 760, 700

BG = (13, 19, 30)
PANEL = (24, 34, 49)
HOVER = (39, 55, 73)
LINE = (55, 72, 92)
WHITE = (237, 243, 250)
MUTED = (156, 174, 194)
GREEN = (45, 212, 191)
RED = (248, 113, 113)

CHOICES = {
    "width": [10, 12, 15, 20, 25, 30],
    "height": [10, 12, 15, 20, 25, 30],
    "fps": [5, 10, 15, 20, 30, 60],
    "sessions": [1, 10, 100, 1000, 10000, 100000],
    "max_steps": [500, 1000, 5000, 10000],
}

ROWS = [
    ("Agent", "agent", 115),
    ("Modèle à charger", "model", 175),
    ("Largeur du plateau", "width", 235),
    ("Hauteur du plateau", "height", 295),
    ("Vitesse (FPS)", "fps", 355),
    ("Sessions", "sessions", 415),
    ("Mouvements max.", "max_steps", 475),
]


def font(size):
    return pygame.font.Font(None, size)


def text(screen, value, x, y, size=24, color=WHITE):
    image = font(size).render(str(value), True, color)
    screen.blit(image, (x, y))


def button(screen, rect, value, active=False):
    color = HOVER if active else PANEL
    pygame.draw.rect(
        screen, color, rect, border_radius=9
    )
    pygame.draw.rect(
        screen, LINE, rect, 1, border_radius=9
    )

    image = font(25).render(str(value), True, WHITE)
    screen.blit(image, image.get_rect(center=rect.center))


def model_files(agent_type):
    if agent_type == "dqn":
        files = list(Path("models").glob("*.pth"))
        preferred = Path("models/dqn-best.pth")
    else:
        files = (
            list(Path("models").glob("*.pkl"))
            + list(Path("train").glob("*.pkl"))
        )
        preferred = Path("train/1000000-v6.pkl")

    files = [p for p in files if p.is_file()]
    files.sort(
        key=lambda p: (
            p == preferred,
            p.stat().st_mtime,
        ),
        reverse=True,
    )
    return [str(p) for p in files] + [None]


def next_filename(agent_type):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    extension = ".pth" if agent_type == "dqn" else ".pkl"
    base = Path("models") / f"gui-{agent_type}-{stamp}"

    path = Path(str(base) + extension)
    number = 1

    while path.exists():
        path = Path(f"{base}-{number}{extension}")
        number += 1

    return str(path)


def show_lobby(args):
    """Renvoie les paramètres choisis, ou None si fermeture."""
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Learn2Slither - Configuration")
    clock = pygame.time.Clock()

    args.visual = "on"
    args.dontlearn = True

    models = model_files(args.agent)
    selected = (
        models.index(args.load)
        if args.load in models
        else 0
    )

    message = ""
    running = True

    while running:
        mouse = pygame.mouse.get_pos()

        screen.fill(BG)

        text(screen, "LEARN2SLITHER", 32, 24, 45)
        text(
            screen,
            "Configuration de la partie",
            34, 73, 24, MUTED,
        )

        for label, key, y in ROWS:
            text(screen, label, 36, y + 9, 25)

            left = pygame.Rect(405, y, 43, 42)
            middle = pygame.Rect(455, y, 235, 42)
            right = pygame.Rect(697, y, 43, 42)

            button(screen, left, "<", left.collidepoint(mouse))
            button(screen, right, ">", right.collidepoint(mouse))

            if key == "agent":
                value = args.agent.upper()
            elif key == "model":
                path = models[selected]
                value = Path(path).name if path else "Aucun"
                if len(value) > 23:
                    value = value[:20] + "..."
            else:
                value = str(getattr(args, key))

            button(screen, middle, value)

        checks = [
            ("Apprentissage actif", not args.dontlearn, 535),
            ("Mode pas à pas", args.step_by_step, 570),
        ]

        for label, checked, y in checks:
            rect = pygame.Rect(36, y, 24, 24)
            pygame.draw.rect(
                screen,
                GREEN if checked else PANEL,
                rect,
                border_radius=5,
            )
            pygame.draw.rect(
                screen, LINE, rect, 1, border_radius=5
            )
            if checked:
                text(screen, "X", 42, y + 1, 22, BG)
            text(screen, label, 71, y + 2, 23)

        play_rect = pygame.Rect(36, 615, 515, 55)
        quit_rect = pygame.Rect(565, 615, 175, 55)

        pygame.draw.rect(
            screen, GREEN, play_rect, border_radius=11
        )
        text(screen, "JOUER", 247, 628, 33, BG)

        button(
            screen, quit_rect, "QUITTER",
            quit_rect.collidepoint(mouse),
        )

        if message:
            text(screen, message, 36, 594, 19, RED)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None

                if event.key == pygame.K_RETURN:
                    if args.dontlearn and models[selected] is None:
                        message = "Choisis un modèle pour jouer sans apprendre."
                    else:
                        running = False

            if event.type != pygame.MOUSEBUTTONDOWN:
                continue

            if event.button != 1:
                continue

            pos = event.pos
            message = ""

            if quit_rect.collidepoint(pos):
                return None

            if play_rect.collidepoint(pos):
                if args.dontlearn and models[selected] is None:
                    message = "Choisis un modèle pour jouer sans apprendre."
                else:
                    running = False
                continue

            if pygame.Rect(36, 535, 340, 26).collidepoint(pos):
                args.dontlearn = not args.dontlearn
                continue

            if pygame.Rect(36, 570, 340, 26).collidepoint(pos):
                args.step_by_step = not args.step_by_step
                continue

            for _, key, y in ROWS:
                direction = 0

                if pygame.Rect(405, y, 43, 42).collidepoint(pos):
                    direction = -1

                if pygame.Rect(697, y, 43, 42).collidepoint(pos):
                    direction = 1

                if direction == 0:
                    continue

                if key == "agent":
                    args.agent = (
                        "dqn" if args.agent == "qtable" else "qtable"
                    )
                    models = model_files(args.agent)
                    selected = 0

                elif key == "model":
                    selected = (
                        selected + direction
                    ) % len(models)

                else:
                    choices = CHOICES[key]
                    current = getattr(args, key)

                    if current in choices:
                        index = choices.index(current)
                    else:
                        index = 0

                    index = (index + direction) % len(choices)
                    setattr(args, key, choices[index])

                break

        clock.tick(30)

    args.load = models[selected]

    if args.dontlearn:
        args.save = None
    else:
        args.save = next_filename(args.agent)

    return args
