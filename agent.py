import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import random

KeyType = Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]
ValueType = Tuple[float, float, float, float]


class Agent:
    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.9,
        eps_start: float = 0.2,  # random au début
        eps_end: float = 0.0,  # random minimum
        eps_decay_steps: int = 200_000,  # vitesse de décroissance
        seed: Optional[int] = None,
    ):
        self.registre: Dict[KeyType, ValueType] = {}
        self.lastState: Optional[KeyType] = None
        self.lastChoice: int = 0

        self.alpha = alpha
        self.gamma = gamma

        # epsilon-greedy décroissant
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = max(1, eps_decay_steps)
        self.step_count = 0

        if seed is not None:
            random.seed(seed)

    def epsilon(self) -> float:
        """
        Décroissance linéaire:
        eps = eps_start -> eps_end sur eps_decay_steps steps.
        """
        t = min(1.0, self.step_count / self.eps_decay_steps)
        return self.eps_start + (self.eps_end - self.eps_start) * t

    def register(self, state: KeyType) -> int:
        self.lastState = state
        if state not in self.registre:
            self.registre[state] = (1.0, 1.0, 1.0, 1.0)

        eps = self.epsilon()

        # exploration
        if random.random() < eps:
            self.lastChoice = random.randint(0, 3)
            return self.lastChoice

        # exploitation
        best_i = 0
        best_v = -(10**18)
        for i, v in enumerate(self.registre[state]):
            if v > best_v:
                best_v = v
                best_i = i
        self.lastChoice = best_i
        return self.lastChoice

    def changeLast(self, reward: float, next_state: KeyType, done: bool) -> None:
        if self.lastState is None:
            return

        if next_state not in self.registre:
            self.registre[next_state] = (0.0, 0.0, 0.0, 0.0)

        v0, v1, v2, v3 = self.registre[self.lastState]
        qs = [v0, v1, v2, v3]

        a = self.lastChoice
        q_sa = qs[a]
        max_next = max(self.registre[next_state])

        target = reward if done else (reward + self.gamma * max_next)
        qs[a] = q_sa + self.alpha * (target - q_sa)

        self.registre[self.lastState] = (qs[0], qs[1], qs[2], qs[3])
        self.step_count += 1

    def getRegistre(self):
        return self.registre

    # ---------- SAVE / LOAD ----------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "registre": self.registre,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "eps_start": self.eps_start,
            "eps_end": self.eps_end,
            "eps_decay_steps": self.eps_decay_steps,
            "step_count": self.step_count,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path, strict: bool = False) -> None:
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)

        self.registre = payload["registre"]
        self.step_count = payload.get("step_count", 0)

        if strict:
            self.alpha = payload["alpha"]
            self.gamma = payload["gamma"]
            self.eps_start = payload["eps_start"]
            self.eps_end = payload["eps_end"]
            self.eps_decay_steps = payload["eps_decay_steps"]
