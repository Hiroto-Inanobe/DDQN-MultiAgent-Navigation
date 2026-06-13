import numpy as np
import random

# ================================
# Map
# ================================
def generate_map():
    str_map = [
        "WWWWWWWWWWWWWWWWWWWWWWWWWWWW",
        "W  TTTTTTTTTT  TTTTTTTTTT  W",
        "W                          W",
        "W                          W",
        "W  CCCCC        CCCCC      W",
        "W                          W",
        "W                          W",
        "WWWWWWWWWWW||====||||====||W",
        "W MMMMM W                  W",
        "W       W                  W",
        "W       W      CCCCCC      W",
        "W              CCCCCC      W",
        "W                          W",
        "W EEEEE                    W",
        "WWWWWWWWWWWWWWWWWWWWWWWWWWWW"
    ]

    grid = np.array([list(row) for row in str_map], dtype=str)

    exit_cells, start_cells, wall_cells = [], [], []

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x] == "E":
                exit_cells.append((y, x))
            elif grid[y, x] == "T":
                start_cells.append((y, x))
            elif grid[y, x] in ["W", "C", "M", "|"]:
                wall_cells.append((y, x))

    return grid, exit_cells, start_cells, wall_cells


# ================================
# Visibility
# ================================
def is_visible(cy, cx, y, x, grid):
    dy = y - cy
    dx = x - cx
    steps = max(abs(dy), abs(dx))

    for k in range(1, steps + 1):
        iy = cy + int(round(dy * k / steps))
        ix = cx + int(round(dx * k / steps))
        if grid[iy, ix] in ["W", "C", "M"]:
            return False
    return True


# ================================
# Agent
# ================================
class Agent:
    def __init__(self, agent_id, start_pos):
        self.id = agent_id
        self.start_pos = start_pos
        self.pos = list(start_pos)
        self.done = False

    def reset(self):
        self.pos = list(self.start_pos)
        self.done = False


# ================================
# Environment
# ================================
class MultiAgentEnv:
    def __init__(self, grid, exit_cells, start_cells, wall_cells,
                 n_agents, obs_size,
                 goal_reward, dist_reward,
                 collision_penalty, error_penalty, step_penalty):

        self.grid = grid
        self.exit_cells = exit_cells
        self.start_cells = start_cells
        self.wall_cells = wall_cells

        self.n_agents = n_agents
        self.obs_size = obs_size

        self.goal_reward = goal_reward
        self.dist_reward = dist_reward
        self.collision_penalty = collision_penalty
        self.error_penalty = error_penalty
        self.step_penalty = step_penalty

        self.action_list = [
            (-1,-1), (-1,0), (-1,1),
            (0,-1), (0,0), (0,1),
            (1,-1), (1,0), (1,1)
        ]

        self.reset()

    def reset(self):
        starts = random.sample(self.start_cells, self.n_agents)
        self.agents = [Agent(i, starts[i]) for i in range(self.n_agents)]
        return self.get_all_observations()

    def get_observation(self, agent):
        obs = np.full((5, self.obs_size, self.obs_size), -1.0, dtype=np.float32)

        cy, cx = agent.pos
        half = self.obs_size // 2
        H, W = self.grid.shape

        for i in range(self.obs_size):
            for j in range(self.obs_size):
                y = cy + (i - half)
                x = cx + (j - half)

                if not (0 <= y < H and 0 <= x < W):
                    continue

                if not is_visible(cy, cx, y, x, self.grid):
                    continue

                obs[:, i, j] = 0.0

                for other in self.agents:
                    if not other.done and other.id != agent.id and other.pos == [y, x]:
                        obs[0, i, j] = 1.0

                if self.grid[y, x] == "|":
                    obs[1, i, j] = 0.3
                elif self.grid[y, x] == "C":
                    obs[1, i, j] = 0.6
                elif self.grid[y, x] == "M":
                    obs[1, i, j] = 0.8
                elif self.grid[y, x] == "W":
                    obs[1, i, j] = 1.0

                if self.grid[y, x] == "=":
                    obs[2, i, j] = 1.0
                if self.grid[y, x] == "T":
                    obs[3, i, j] = 1.0
                if self.grid[y, x] == "E":
                    obs[4, i, j] = 1.0

        return obs

    def get_all_observations(self):
        return [self.get_observation(a) for a in self.agents]

    def step(self, actions):
        rewards = [0.0] * self.n_agents
        dones = [False] * self.n_agents

        next_positions = [a.pos[:] for a in self.agents]

        collision_agents = []
        error_agents = []

        for ag, a in zip(self.agents, actions):
            if ag.done:
                continue

            dy, dx = self.action_list[a]
            ny, nx = ag.pos[0] + dy, ag.pos[1] + dx

            rewards[ag.id] += self.step_penalty

            if not (0 <= ny < self.grid.shape[0] and 0 <= nx < self.grid.shape[1]) \
               or self.grid[ny, nx] in ["W","C","M","|"]:
                rewards[ag.id] += self.error_penalty
                error_agents.append(ag.id)
                continue

            py, px = ag.pos
            prev_dist = min(abs(py-gy)+abs(px-gx) for gy,gx in self.exit_cells)
            new_dist = min(abs(ny-gy)+abs(nx-gx) for gy,gx in self.exit_cells)

            if new_dist < prev_dist:
                rewards[ag.id] += self.dist_reward
                next_positions[ag.id] = [ny, nx]

        # collision
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                if next_positions[i] == next_positions[j]:
                    rewards[i] += self.collision_penalty
                    rewards[j] += self.collision_penalty
                    collision_agents.extend([i,j])

        for i in range(self.n_agents):
            self.agents[i].pos = next_positions[i]

        for ag in self.agents:
            if tuple(ag.pos) in self.exit_cells:
                rewards[ag.id] += self.goal_reward
                ag.done = True
                dones[ag.id] = True

        return self.get_all_observations(), rewards, dones, collision_agents, error_agents
