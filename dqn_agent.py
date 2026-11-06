import random
from collections import deque
from pathlib import Path

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.layers(x)


class DQNAgent:
    def __init__(self, seed=None):
        torch.set_num_threads(2)

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.device = torch.device("cpu")
        self.policy = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=0.0003
        )

        self.memory = deque(maxlen=20000)
        self.batch_size = 64
        self.gamma = 0.99
        self.step_count = 0
        self.update_every = 4
        self.target_every = 1000
        self.warmup = 256

        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay_steps = 200000

        self.last_state = None
        self.last_action = None
        self.last_loss = None

    @staticmethod
    def encode(state):
        """Convertit les 16 valeurs de vision en entrée numérique."""
        return tuple(
            float(value) / 4.0
            for direction in state
            for value in direction
        )

    @staticmethod
    def allowed_actions(state):
        """Évite les obstacles immédiats visibles."""
        actions = [
            i for i, (wall, green, red, body)
            in enumerate(state)
            if wall != 1 and body != 1
        ]
        return actions if actions else [0, 1, 2, 3]

    def epsilon(self):
        progress = min(
            1.0, self.step_count / self.eps_decay_steps
        )
        return self.eps_start + (
            self.eps_end - self.eps_start
        ) * progress

    def choose(self, state, explore=False):
        allowed = self.allowed_actions(state)

        if explore and random.random() < self.epsilon():
            return random.choice(allowed)

        values = torch.tensor(
            [self.encode(state)],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            q_values = self.policy(values)[0]

        best_value = max(q_values[a].item() for a in allowed)
        best_actions = [
            a for a in allowed
            if q_values[a].item() == best_value
        ]
        return random.choice(best_actions)

    def register(self, state):
        action = self.choose(state, explore=True)
        self.last_state = state
        self.last_action = action
        return action

    def changeLast(self, reward, next_state, done):
        if self.last_state is None:
            return

        # Récompenses normalisées pour stabiliser le DQN.
        scaled_reward = max(
            -10.0, min(10.0, float(reward) / 10.0)
        )

        self.memory.append((
            self.encode(self.last_state),
            self.last_action,
            scaled_reward,
            self.encode(next_state),
            bool(done),
            self.allowed_actions(next_state),
        ))

        self.step_count += 1
        self.last_state = None
        self.last_action = None

        if (
            len(self.memory) >= self.warmup
            and self.step_count % self.update_every == 0
        ):
            self.learn()

        if self.step_count % self.target_every == 0:
            self.target.load_state_dict(
                self.policy.state_dict()
            )

    def learn(self):
        batch = random.sample(self.memory, self.batch_size)

        states = torch.tensor(
            [item[0] for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            [item[1] for item in batch],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.tensor(
            [item[2] for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.tensor(
            [item[3] for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [item[4] for item in batch],
            dtype=torch.float32,
            device=self.device,
        )

        q_values = self.policy(states)
        current_q = q_values.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_q = self.target(next_states)

            mask = torch.zeros_like(
                next_q, dtype=torch.bool
            )
            for index, item in enumerate(batch):
                mask[index, item[5]] = True

            next_q = next_q.masked_fill(
                ~mask, float("-inf")
            )
            best_next = next_q.max(dim=1).values

            expected_q = rewards + (
                1.0 - dones
            ) * self.gamma * best_next

        loss = nn.functional.smooth_l1_loss(
            current_q, expected_q
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy.parameters(), 10.0
        )
        self.optimizer.step()

        self.last_loss = loss.item()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "memory": list(self.memory),
            "step_count": self.step_count,
            "eps_start": self.eps_start,
            "eps_end": self.eps_end,
            "eps_decay_steps": self.eps_decay_steps,
            "gamma": self.gamma,
        }, path)

    def load(self, path):
        payload = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )

        self.policy.load_state_dict(payload["policy"])
        self.target.load_state_dict(payload["target"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.memory = deque(
            payload.get("memory", []),
            maxlen=20000,
        )
        self.step_count = payload["step_count"]
        self.eps_start = payload["eps_start"]
        self.eps_end = payload["eps_end"]
        self.eps_decay_steps = payload["eps_decay_steps"]
        self.gamma = payload["gamma"]
        self.target.eval()

        self.last_state = None
        self.last_action = None
