import random
from typing import Dict, Tuple

KeyType = Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]
ValueType = Tuple[float, float, float, float]


class Agent:
    registre: Dict[KeyType, ValueType] = {}
    lastState: KeyType
    lastChoice: int

    # paramètres Q-learning
    alpha = 0.2  # vitesse d'apprentissage
    gamma = 0.9  # importance du futur

    def register(self, state):
        self.lastState = state
        if state not in self.registre:
            self.registre[state] = (1, 1, 1, 1)

        if random.random() < 0.2:
            self.lastChoice = random.randint(0, 3)
            return self.lastChoice

        elMax = -1000
        elI = 0
        for i, el in enumerate(self.registre[state]):
            if el > elMax:
                elMax = el
                elI = i
        self.lastChoice = elI
        return self.lastChoice

    # ✅ Q-learning update: a besoin du prochain état
    def changeLast(self, reward, next_state):
        # crée next_state si absent
        if next_state not in self.registre:
            self.registre[next_state] = (1, 1, 1, 1)

        v0, v1, v2, v3 = self.registre[self.lastState]
        qs = [v0, v1, v2, v3]

        a = self.lastChoice
        q_sa = qs[a]
        max_next = max(self.registre[next_state])

        # Q(s,a) <- Q(s,a) + alpha * (reward + gamma*max_next - Q(s,a))
        qs[a] = q_sa + self.alpha * (reward + self.gamma * max_next - q_sa)

        self.registre[self.lastState] = (qs[0], qs[1], qs[2], qs[3])

    def getRegistre(self):
        return self.registre
