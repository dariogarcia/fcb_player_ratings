import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from efficient_apriori import apriori
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from time_gif import create_accumulated_votes_animation
from mydt import run_decision_tree, run_decision_tree_cv
from myclustering import run_hierarchical_clustering, run_kmeans


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
    # This matrix contains just a boolean per player. Was it voted or not.
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
    # This matrix contains the total of votes per player
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

import numpy as np
import csv

def derive_pos_sum_votes(data, column_names, csv_file_path):
    num_rows, num_cols = data.shape
    num_players = num_cols // 3
    position_mapping = {}
    with open(csv_file_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            column_name, position = row[0], row[1]
            position_mapping[column_name] = position
    # Create a dictionary to store the aggregated sums for each position
    unique_positions = set(position_mapping.values())
    aggregated_sums = {position: np.zeros((num_rows,), dtype=int) for position in unique_positions}
    for i in range(num_players):
        column_name = column_names[i * 3][:-2]
        position = position_mapping.get(column_name, "")
        transformed_column = data[:, i * 3] * 1 + data[:, i * 3 + 1] * 3 + data[:, i * 3 + 2] * 5
        aggregated_sums[position] = aggregated_sums[position] + transformed_column
    # Convert the dictionary of aggregated sums to a numpy array with the correct data type
    pos_sum_matrix = np.column_stack([aggregated_sums[position].astype(int) for position in unique_positions])
    return pos_sum_matrix, list(unique_positions)

def to_transactions(data, item_names):
    #Asumes matrix is binary. This makes player_bool and player_sum identical
    if not np.all(np.logical_or(data == 0, data == 1)):
        warnings.warn("The data matrix is not binary. Non-binary values will be treated as 1's.")
    # transforms a data matrix into a list of transactions
    transactions = []
    for row in data:
        transaction = [item_names[i] for i, item in enumerate(row) if item >= 1]
        transactions.append(tuple(transaction))
    return transactions

def player_rankings(data, column_names):
    num_players = len(column_names)
    player_points = np.sum(data, axis=0)  # Calculate the total points for each player
    player_ranking = np.argsort(player_points)[::-1]  # Get the indices that would sort the players in descending order

    print("Player Rankings:")
    for rank, player_idx in enumerate(player_ranking, start=1):
        player_name = column_names[player_idx]
        points = player_points[player_idx]
        print(f"Rank {rank}: {player_name} - Total Points: {points}")
    return

# Itemsets
def run_apriori(data_columns, row_labels, min_support = 0.34, min_confidence = 0.5):
    transactions = to_transactions(data_columns, row_labels)
    #TODO: Check if matrix is binary
    itemsets, rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence, verbosity=2)
    print("Transactions:")
    print(transactions)
    print("Frequent Itemsets:")
    print(itemsets)
    print("\nAssociation Rules:")
    for rule in rules:
        print(rule)
    return

def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform data analysis on FCB player ratings.')
    parser.add_argument('file_path', type=str, help='Path to the input CSV file')
    parser.add_argument('--label', type=str, choices=['date', 'outcome', 'competition'], default=None,
                        help='Column to use as label (default: None)')
    parser.add_argument('--data', type=str, choices=['original','player_bool', 'player_sum', 'position_sum'], default='player_bool',
                        help='Column to use as data (default: player_bool)')
                        #original: Each player as three features 1p, 3p and 5p
                        #player_bool: Either the player gets one or more votes, or it did not.
                        #player_sum: Each player has one feature: total points per game
                        #position_sum: points by players in GK, DF, MF and FW positions are aggregated
    return parser.parse_args()


# Main
if __name__ == "__main__":
    args = parse_arguments()
    file_path = args.file_path
    original_data, original_item_names = read_data_matrix(file_path)
    label_column = None
    #Define label
    if args.label == 'date':
        label_column = original_data[:, 0]
    elif args.label == 'outcome':
        label_column = original_data[:, 1]
    elif args.label == 'competition':
        label_column = original_data[:, 2]
    #Define data
    if args.data == 'player_bool':
        data_columns, row_labels = derive_voted(original_data[:,3:], original_item_names[3:])
    elif args.data == 'player_sum':
        data_columns, row_labels = derive_sum_votes(original_data[:,3:], original_item_names[3:])
    elif args.data == 'original':
        data_columns, row_labels = original_data[:,3], original_item_names[3:]
    if args.data == 'position_sum':
        data_columns, row_labels = derive_pos_sum_votes(original_data[:,3:], original_item_names[3:], '../data/players_raw.csv')
    
    #Run Apriori
    #run_apriori(data_columns, row_labels)
    
    #Run K-means
    #run_kmeans(data_columns, 3, label_column)
    run_hierarchical_clustering(data_columns, 3, label_column)
    #Run Decision Tree
    #run_decision_tree(data_columns, label_column, row_labels, max_depth=5)
    #run_decision_tree_cv(data_columns, label_column, row_labels, max_depth=5)

    #Create accumulated animation
    #if  args.data != 'original':
    #    create_accumulated_votes_animation(data_columns, row_labels)
    
    #Pring player rankings
    #data_columns, row_labels = derive_sum_votes(original_data[:,3:], original_item_names[3:])
    #player_rankings(data_columns, row_labels)