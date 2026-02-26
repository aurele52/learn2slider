import pickle
from pathlib import Path
from typing import Tuple, Optional
import random
from array import array
from collections import defaultdict

# -------- Types (same observation format as Interpreter.get_state) --------
DirFeat = Tuple[int, int, int, int]  # (wall, green, red, body)
StateType = Tuple[DirFeat, DirFeat, DirFeat, DirFeat]  # (up, right, down, left)

# -------- Symmetries (URDL indices: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT) --------
# IMPORTANT:
# - _apply_dir_map uses dir_map as "new_index -> old_index"
# - so ROT_DIR[k][new_dir] = old_dir
ROT_DIR = (
    (0, 1, 2, 3),  # 0°
    (3, 0, 1, 2),  # 90°  (new UP comes from old LEFT, etc.)
    (2, 3, 0, 1),  # 180°
    (1, 2, 3, 0),  # 270°
)

# Mirror across vertical axis: swap LEFT<->RIGHT.
# This is also used as "new_index -> old_index"
MIRROR_DIR = (0, 3, 2, 1)

# -------- Packing (base-5 for 16 bins, because your bins are in 0..4) --------
BASE = 5
NFEATS = 16
POW = [BASE**i for i in range(NFEATS)]


def _apply_dir_map(
    state_urdl: StateType, dir_map: Tuple[int, int, int, int]
) -> StateType:
    return (
        state_urdl[dir_map[0]],
        state_urdl[dir_map[1]],
        state_urdl[dir_map[2]],
        state_urdl[dir_map[3]],
    )


def _transform_state(
    state_urdl: StateType, rot_k: int, mirror: bool = False
) -> StateType:
    """
    Returns transformed state in canonical frame.
    We apply mirror first, then rotation (same order as earlier discussion).
    """
    s = state_urdl
    if mirror:
        s = _apply_dir_map(s, MIRROR_DIR)
    s = _apply_dir_map(s, ROT_DIR[rot_k])
    return s


def _canon_to_env_action(a_can: int, rot_k: int, mirror: bool) -> int:
    """
    Map an action chosen in canonical frame back to env frame.

    Because ROT_DIR and MIRROR_DIR are defined as "new -> old" (new_index maps to old_index),
    and _transform_state applies mirror first then rotation, the combined new->old mapping is:

      old = (mirror ? MIRROR_DIR[ old_after_rot ] : old_after_rot)
      old_after_rot = ROT_DIR[rot_k][new]

    So:
      a_env = ROT_DIR[rot_k][a_can]
      if mirror: a_env = MIRROR_DIR[a_env]
    """
    a_env = ROT_DIR[rot_k][a_can]
    if mirror:
        a_env = MIRROR_DIR[a_env]
    return a_env


def _pack_state_16_base5(state_urdl: StateType) -> int:
    # flatten order: U feats then R then D then L, each feats=(wall,green,red,body)
    key = 0
    i = 0
    for d in range(4):
        w, g, r, b = state_urdl[d]
        key += w * POW[i]
        i += 1
        key += g * POW[i]
        i += 1
        key += r * POW[i]
        i += 1
        key += b * POW[i]
        i += 1
    return key


def _canonical_pack_key(
    state_urdl: StateType, use_mirror: bool = False
) -> tuple[int, tuple[int, bool]]:
    """
    Returns:
      - best_key: packed int of the canonicalized state
      - best_params: (rot_k, mirror) that produced that canonical form
    """
    best_key: Optional[int] = None
    best_params: tuple[int, bool] = (0, False)

    mirrors = (False, True) if use_mirror else (False,)
    for mirror in mirrors:
        for rot_k in range(4):
            s2 = _transform_state(state_urdl, rot_k, mirror)
            k2 = _pack_state_16_base5(s2)
            if best_key is None or k2 < best_key:
                best_key = k2
                best_params = (rot_k, mirror)

    assert best_key is not None
    return best_key, best_params


class Agent:
    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.9,
        eps_start: float = 0.2,
        eps_end: float = 0.0,
        eps_decay_steps: int = 200_000,
        seed: Optional[int] = None,
        use_mirror: bool = True,  # rotations only by default; set True to enable reflections too
    ):
        # Q-table: key(int) -> mutable array('f',4)
        self.registre = defaultdict(lambda: array("f", [0.0, 0.0, 0.0, 0.0]))

        self.lastKey: Optional[int] = None
        self.lastChoice_can: int = 0

        self.alpha = alpha
        self.gamma = gamma

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = max(1, eps_decay_steps)
        self.step_count = 0

        self.use_mirror = use_mirror

        if seed is not None:
            random.seed(seed)

    def epsilon(self) -> float:
        t = min(1.0, self.step_count / self.eps_decay_steps)
        return self.eps_start + (self.eps_end - self.eps_start) * t

    def register(self, state_env: StateType) -> int:
        """
        Input: state in env frame (URDL).
        Returns: action in env frame (0..3).
        Internally:
          - canonicalize+pack state -> key
          - choose action in canonical frame
          - map back to env frame before returning
        """
        key, (rot_k, mirror) = _canonical_pack_key(
            state_env, use_mirror=self.use_mirror
        )
        state_can = _transform_state(state_env, rot_k, mirror)

        self.lastKey = key  # store packed canonical key

        eps = self.epsilon()

        # allowed_actions computed in CANONICAL frame (important!)
        safe_actions = []
        for a_can, dirfeat in enumerate(state_can):
            wall_bin, green_bin, red_bin, body_bin = dirfeat
            if wall_bin == 1 or body_bin == 1:
                continue
            safe_actions.append(a_can)

        allowed_can = safe_actions if safe_actions else [0, 1, 2, 3]

        # epsilon-greedy in canonical frame
        if random.random() < eps:
            a_can = random.choice(allowed_can)
        else:
            q = self.registre[key]
            best_v = max(q[a] for a in allowed_can)
            best_actions = [a for a in allowed_can if q[a] == best_v]
            a_can = random.choice(best_actions)

        self.lastChoice_can = a_can

        # map canonical action back to env action (Option A, direct mapping)
        a_env = _canon_to_env_action(a_can, rot_k, mirror)
        return a_env

    def changeLast(self, reward: float, next_state_env: StateType, done: bool) -> None:
        """
        Update Q using the CANONICAL action stored in lastChoice_can.
        next_state is canonicalized+packed too (independent canonicalization is fine).
        """
        if self.lastKey is None:
            return

        next_key, _ = _canonical_pack_key(next_state_env, use_mirror=self.use_mirror)

        q = self.registre[self.lastKey]
        a = self.lastChoice_can
        q_sa = q[a]

        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.registre[next_key])

        q[a] = q_sa + self.alpha * (target - q_sa)
        self.step_count += 1

    def getRegistre(self):
        return self.registre

    # ---------- SAVE / LOAD ----------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # defaultdict(lambda: ...) not safely picklable due to lambda -> save as plain dict
        registre_plain = {k: v for k, v in self.registre.items()}

        payload = {
            "registre": registre_plain,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "eps_start": self.eps_start,
            "eps_end": self.eps_end,
            "eps_decay_steps": self.eps_decay_steps,
            "step_count": self.step_count,
            "use_mirror": self.use_mirror,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path, strict: bool = False) -> None:
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)

        loaded = payload["registre"]

        # rebuild defaultdict factory
        self.registre = defaultdict(lambda: array("f", [0.0, 0.0, 0.0, 0.0]))
        for k, v in loaded.items():
            if isinstance(v, array):
                self.registre[k] = v
            else:
                self.registre[k] = array("f", v)

        self.step_count = payload.get("step_count", 0)
        self.use_mirror = payload.get("use_mirror", True)

        if strict:
            self.alpha = payload["alpha"]
            self.gamma = payload["gamma"]
            self.eps_start = payload["eps_start"]
            self.eps_end = payload["eps_end"]
            self.eps_decay_steps = payload["eps_decay_steps"]
