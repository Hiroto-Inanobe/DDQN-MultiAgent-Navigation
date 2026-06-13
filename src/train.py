import numpy as np
import torch
import matplotlib.pyplot as plt

from environment import generate_map, MultiAgentEnv
from ddqn_agent import DDQNAgent
from visualization import animate_episode, show_animation

# =========================
# Hyperparameters
# =========================
n_agents = 20
obs_size = 7

n_episodes = 20000
max_steps = 1000

alpha = 2.5e-4
gamma = 0.98
batch_size = 128
buffer_size = 150000
polyak = 0.999

eps_start = 1.0
eps_end = 0.1
eps_decay = 1200

goal_reward = 300
dist_reward = 0.1
collision_penalty = -2.0
error_penalty = -0.5
step_penalty = -0.1


# =========================
# Init
# =========================
grid, exit_cells, start_cells, wall_cells = generate_map()

env = MultiAgentEnv(
    grid, exit_cells, start_cells, wall_cells,
    n_agents, obs_size,
    goal_reward, dist_reward,
    collision_penalty, error_penalty, step_penalty
)

agent = DDQNAgent(
    in_ch=5,
    obs_size=obs_size,
    out_dim=9,
    alpha=alpha,
    gamma=gamma,
    batch_size=batch_size,
    buffer_size=buffer_size,
    polyak=polyak
)

reward_log = []


# =========================
# Training loop
# =========================
for ep in range(n_episodes):

    obs = env.reset()

    eps = max(eps_end, eps_start * np.exp(-ep / eps_decay))

    episode_positions = []

    total_reward = 0

    for t in range(max_steps):

        actions = [agent.act(obs[i], eps) for i in range(n_agents)]

        next_obs, rewards, dones, _, _ = env.step(actions)

        for i in range(n_agents):
            agent.buffer.push(obs[i], actions[i], rewards[i], next_obs[i], dones[i])

        agent.update()

        obs = next_obs
        total_reward += sum(rewards)

        episode_positions.append([ag.pos[:] for ag in env.agents])

        if all(dones):
            break

    reward_log.append(total_reward / n_agents)

    if ep % 500 == 0:
        print(f"Episode {ep} | Reward {reward_log[-1]:.2f}")

        ani = animate_episode(env, episode_positions)
        display(show_animation(ani))


# =========================
# Plot
# =========================
plt.plot(reward_log)
plt.title("Mean Reward")
plt.show()
