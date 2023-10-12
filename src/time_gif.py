import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_accumulated_votes_animation(data, labels, fps=0.1):
    # Data for the number of games and players
    num_games, num_players = data.shape
    players = range(1, num_players + 1)
    accumulated_data = np.zeros_like(data)
    fig, ax = plt.subplots(figsize=(10,6))
    def update(frame):
        ax.clear()
        ax.set_title(f"Accumulated Points Over Games (Game {frame + 1})")
        ax.set_xlabel("Players")
        ax.set_ylabel("Accumulated Points")
        # Accumulate points up to the current game
        accumulated_data[frame, :] = np.sum(data[:frame+1, :], axis=0)
        ax.bar(players, accumulated_data[frame, :])
        ax.set_xticks(players)
        ax.set_xticklabels(labels, rotation=45, ha='right')
    ani = FuncAnimation(fig, update, frames=range(num_games), repeat=False, blit=False, interval=1000/fps)
    ani.save('accumulated_points.gif', writer='pillow', fps=fps)