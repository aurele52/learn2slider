"""Sauvegardes et checkpoints des agents."""

from pathlib import Path


def save_checkpoint(agent, args, episode):
    """Sauvegarde périodique pendant l'entraînement DQN."""
    if args.agent != "dqn":
        return

    if args.dontlearn or not args.save:
        return

    if episode % args.checkpoint_every != 0:
        return

    directory = Path(args.save).parent
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"dqn-step-{agent.step_count}.pth"

    if path.exists():
        print(f"[CHECKPOINT] Déjà présent : {path}")
        return

    agent.save(path)
    print(f"[CHECKPOINT] {path}", flush=True)


def save_final(agent, args):
    """Sauvegarde le modèle à la fin de l'exécution."""
    if not args.save:
        return

    if args.dontlearn:
        print("Mode sans apprentissage : sauvegarde ignorée.")
        return

    agent.save(args.save)
    print(f"Modèle sauvegardé : {args.save}")
