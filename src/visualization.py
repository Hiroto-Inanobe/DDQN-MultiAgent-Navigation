import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from IPython.display import HTML


def animate_episode(env, episode_positions,
                    collision_dict=None,
                    error_dict=None,
                    goal_dict=None,
                    ep_id=0):

    fig, ax = plt.subplots(figsize=(6, 6))

    # static map
    for y in range(env.grid.shape[0]):
        for x in range(env.grid.shape[1]):
            c = env.grid[y, x]
            if c == "W":
                ax.scatter(x, y, c="black", s=100, marker="s")
            elif c == "C":
                ax.scatter(x, y, c="chocolate", s=100, marker="s")
            elif c == "M":
                ax.scatter(x, y, c="indianred", s=100, marker="s")
            elif c == "|":
                ax.scatter(x, y, c="gray", s=100, marker="s")
            elif c == "=":
                ax.scatter(x, y, c="lightgray", s=100, marker="s")
            elif c == "T":
                ax.scatter(x, y, c="yellow", s=100, marker="s")
            elif c == "E":
                ax.scatter(x, y, c="lime", s=100, marker="s")

    points = ax.scatter([], [], c="purple", s=80)

    ax.set_xlim(-0.5, env.grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, env.grid.shape[0] - 0.5)
    ax.invert_yaxis()

    def update(frame):
        pos = episode_positions[frame]

        xs, ys = [], []
        colors = []

        for i, (y, x) in enumerate(pos):
            ys.append(y)
            xs.append(x)
            colors.append("purple")

        points.set_offsets(np.c_[xs, ys])
        points.set_color(colors)

        ax.set_title(f"Episode {ep_id} Step {frame}")
        return points,

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(episode_positions),
        interval=100,
        blit=True
    )

    plt.close(fig)
    return ani


def show_animation(ani):
    return HTML(ani.to_jshtml())
