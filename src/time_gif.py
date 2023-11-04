import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_accumulated_votes_animation(data, labels, fps=0.3):
    print(labels)
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
        # Sort players by their accumulated points for the current frame
        sorted_players = [p for _, p in sorted(zip(accumulated_data[frame, :], players), reverse=True)]
        # Get the top 3 players
        top_players = sorted_players[:3]
        top_players_colors = ['b', 'g', 'r']
        # Plot bars with different colors for the top 3 players
        for i, player in enumerate(players):
            if player in top_players:
                color = top_players_colors[top_players.index(player)] 
            else:
                color = 'gray'
            ax.bar(player, accumulated_data[frame, player-1], color=color, label=labels[player-1])
        ax.set_xticks(players)
        ax.set_xticklabels(labels, rotation=45, ha='right')
    ani = FuncAnimation(fig, update, frames=range(num_games), repeat=False, blit=False, interval=1000/fps)
    ani.save('../data/accumulated_points.gif', writer='pillow', fps=fps)