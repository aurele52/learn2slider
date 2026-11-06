"""Sélection des actions Q-table sans apprentissage."""

import random
from typing import List

from agent import Agent
from agent import (
    _canonical_pack_key,
    _transform_state,
    _canon_to_env_action,
)

Action = int


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
    # NOTE: agent.registre is a defaultdict, so agent.registre[key] always
    # exists.
    qvals = agent.registre.get(key)

    # allowed actions must be computed in CANONICAL frame
    allowed_can = allowed_actions_no_suicide_from_state(state_can)
    if qvals is None:
        a_can = random.choice(allowed_can)
        return _canon_to_env_action(a_can, rot_k, mirror)

    # greedy in canonical frame
    best_v = max(qvals[a] for a in allowed_can)
    best = [a for a in allowed_can if qvals[a] == best_v]

    a_can = best[0] if deterministic_tiebreak else random.choice(best)

    # map canonical action back to env action (Option A, direct mapping)
    a_env = _canon_to_env_action(a_can, rot_k, mirror)
    return a_env

