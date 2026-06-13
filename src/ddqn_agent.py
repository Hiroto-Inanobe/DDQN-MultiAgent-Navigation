import torch
import torch.nn as nn
import torch.optim as optim
import random

from model import QNetwork
from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDQNAgent:
    def __init__(self, in_ch, obs_size, out_dim,
                 alpha, gamma, batch_size, buffer_size, polyak):

        self.q = QNetwork(in_ch, obs_size, out_dim).to(device)
        self.target = QNetwork(in_ch, obs_size, out_dim).to(device)
        self.target.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=alpha)

        self.buffer = ReplayBuffer(buffer_size)

        self.gamma = gamma
        self.batch_size = batch_size
        self.polyak = polyak

    def act(self, s, eps):
        if random.random() < eps:
            return random.randint(0, 8)

        s = torch.tensor(s[None], dtype=torch.float32, device=device)

        with torch.no_grad():
            return self.q(s).argmax(1).item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        s, a, r, ns, d = self.buffer.sample(self.batch_size)

        s = torch.tensor(s, dtype=torch.float32, device=device)
        ns = torch.tensor(ns, dtype=torch.float32, device=device)
        a = torch.tensor(a, dtype=torch.long, device=device)
        r = torch.tensor(r, dtype=torch.float32, device=device)
        d = torch.tensor(d, dtype=torch.float32, device=device)

        q = self.q(s).gather(1, a[:, None]).squeeze()

        with torch.no_grad():
            na = self.q(ns).argmax(1)
            tq = self.target(ns).gather(1, na[:, None]).squeeze()
            target = r + self.gamma * tq * (1 - d)

        loss = nn.MSELoss()(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Polyak update
        for p, tp in zip(self.q.parameters(), self.target.parameters()):
            tp.data.mul_(self.polyak).add_((1 - self.polyak) * p.data)
