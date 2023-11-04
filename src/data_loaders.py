import csv
import numpy as np
import pandas as pd

# Function to read the original data matrix
def read_data_matrix(file_path):
    try:
        data = pd.read_csv(file_path, delimiter=',', index_col=False)
        data = data.fillna(0)  # Fill NaN values with 0
        return data.values, data.columns
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")

# Derived Matrices
def derive_voted(data, column_names):
    # This matrix contains just a boolean per player. Was it voted or not on each game.
    # Returns both the data matrix, and the shortened header's list.
    num_rows, num_cols = data.shape
    num_players = num_cols // 3
    transformed_matrix = np.zeros((num_rows, num_players), dtype=int)
    player_columns_list = []
    for i in range(num_players):
        player_columns = column_names[i * 3:(i + 1) * 3]
        player_votes = data[:, i * 3:(i + 1) * 3]
        any_nonzero = np.any(player_votes > 0, axis=1)
        transformed_matrix[:, i] = any_nonzero.astype(int)
        #TODO: Fix hardcode of taking the first and slicing
        player_columns_list.append(player_columns[0][:-2])
    return transformed_matrix, player_columns_list

def derive_sum_votes(data, column_names):
    # This matrix contains the total of votes per player per game
    # Returns both the data matrix, and the shortened header's list.
    num_rows, num_cols = data.shape
    num_players = num_cols // 3
    transformed_matrix = np.zeros((num_rows, num_players), dtype=int)
    player_columns_list = []
    for i in range(num_players):
        player_columns = column_names[i * 3:(i + 1) * 3]
        transformed_matrix[:, i] = data[:, i * 3]*1 + data[:, i * 3 + 1]*3 + data[:, i * 3 + 2]*5
        player_columns_list.append(player_columns[0][:-2])
    return transformed_matrix, player_columns_list

def position_order(position):
    # Define the custom sorting order
    order = {'GK': 0, 'DF': 1, 'MF': 2, 'FW': 3}
    return order.get(position, 4)

def derive_pos_sum_votes(data, column_names, csv_file_path):
    # This matrix contains the total of votes per position per game
    # Returns both the data matrix, and the shortened header's list.
    num_rows, num_cols = data.shape
    num_players = num_cols // 3
    position_mapping = {}
    with open(csv_file_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            column_name, position = row[0], row[1]
            position_mapping[column_name] = position
    unique_positions = list(dict.fromkeys(position_mapping.values()))
    unique_positions.sort(key=position_order)
    aggregated_sums = {position: np.zeros((num_rows,), dtype=int) for position in unique_positions}
    for i in range(num_players):
        column_name = column_names[i * 3][:-2] #The -2 strips the "1p/3p/5p" chars from the label
        position = position_mapping.get(column_name, "")
        transformed_column = data[:, i * 3] * 1 + data[:, i * 3 + 1] * 3 + data[:, i * 3 + 2] * 5
        aggregated_sums[position] = aggregated_sums[position] + transformed_column
    pos_sum_matrix = np.column_stack([aggregated_sums[position].astype(int) for position in unique_positions])
    return pos_sum_matrix, unique_positions
    

def player_rankings(data, column_names):
    player_points = np.sum(data, axis=0)
    player_ranking = np.argsort(player_points)[::-1]

    print("Player Rankings:")
    for rank, player_idx in enumerate(player_ranking, start=1):
        print(f"Rank {rank}: {column_names[player_idx]} - Total Points: {player_points[player_idx]}")
    return