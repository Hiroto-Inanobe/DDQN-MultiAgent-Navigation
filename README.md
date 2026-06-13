# Multi-Agent Navigation Learning in Congested Environments Using Double Deep Q-Networks

## Overview

This repository contains the implementation and experimental results of a multi-agent reinforcement learning project based on Double Deep Q-Networks (DDQN).

The objective is to study cooperative navigation behavior in congested environments where multiple agents must reach a common exit while avoiding obstacles and collisions.

The environment is modeled as a partially observable grid world inspired by a railway station layout.

---

## Key Features

- Double Deep Q-Network (DDQN)
- Multi-Agent Reinforcement Learning
- Partial Observation
- Shared Policy Learning
- Collision Avoidance
- Congestion-Aware Navigation

---

## Environment

| Item | Value |
|--------|--------|
| Grid Size | 15 × 28 |
| Number of Agents | 20 |
| Observation Size | 7 × 7 |
| Observation Channels | 5 |
| Action Space | 9 |

### Available Actions

- Up-Left
- Up
- Up-Right
- Left
- Stay
- Right
- Down-Left
- Down
- Down-Right

---

## Observation Representation

Each agent receives a 5-channel local observation:

1. Other agents
2. Terrain information
3. Corridor structure
4. Spawn locations
5. Exit locations

Cells hidden by obstacles are represented as -1.

---

## Neural Network Architecture

| Layer | Output Channels |
|---------|---------|
| Conv2D | 16 |
| Conv2D | 32 |
| Conv2D | 64 |
| Fully Connected | 256 |
| Output Layer | 9 |

Activation Function: ReLU

---

## Training Configuration

| Parameter | Value |
|------------|---------|
| Algorithm | DDQN |
| Optimizer | Adam |
| Learning Rate | 2.5e-4 |
| Discount Factor | 0.98 |
| Batch Size | 128 |
| Replay Buffer Size | 150000 |
| Episodes | 20000 |
| Polyak Coefficient | 0.999 |

---

## Reward Design

| Event | Reward |
|---------|---------|
| Reach Exit | +300 |
| Move Closer to Exit | +0.1 |
| Collision | -2.0 |
| Invalid Move | -0.5 |
| Time Step | -0.1 |

---

## Results

Experimental results demonstrated that agents successfully learned:

- Efficient evacuation behavior
- Collision avoidance
- Cooperative movement
- Navigation through narrow passages

Average rewards increased during training, while collision frequency and evacuation time decreased.

---

## Repository Structure

```text
src/
figures/
thesis/
README.md
requirements.txt
LICENSE
```

---

## Running

```bash
pip install -r requirements.txt

python src/train.py
```

---

## License

Released under the MIT License.

## Author

Anonymous (for privacy)
