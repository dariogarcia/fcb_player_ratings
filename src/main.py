import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from efficient_apriori import apriori
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
    num_rows, num_cols = data.shape
    num_players = num_cols // 3
    transformed_matrix = np.zeros((num_rows, num_players), dtype=int)
    player_columns_list = []

    for i in range(num_players):
        player_columns = column_names[i * 3:(i + 1) * 3]
        player_votes = data[:, i * 3:(i + 1) * 3]
        transformed_matrix[:, i] = np.sum(player_votes, axis=1)  # Sum the votes for each row
        player_columns_list.append(player_columns[0][:-2])

    return transformed_matrix, player_columns_list

def to_transactions(data, item_names):
    # transforms a data matrix into a list of transactions
    transactions = []
    for row in data:
        transaction = [item_names[i] for i, item in enumerate(row) if item == 1]
        transactions.append(tuple(transaction))
    return transactions

# Clustering
def run_kmeans(data, num_clusters, labels=None):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=20)
    clusters = kmeans.fit_predict(data)

    # Define a colormap with a fixed number of colors
    colormap = plt.cm.get_cmap('Dark2_r', num_clusters)
    # Reduce dimensionality for plot
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    cluster_centers = kmeans.cluster_centers_
    # Visualize the clusters
    plt.figure(figsize=(8, 6))

    if labels is not None:
        symbols = ['o', '*', 's', '^', 'D', 'v', 'p', 'H', '+']
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            label_indices = np.where(labels == label)
            scatter = plt.scatter(reduced_data[label_indices, 0], reduced_data[label_indices, 1],
                        label=f'Label {label}', c=colormap(i % num_clusters), marker=symbols[i % len(symbols)])
    
    else:
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap=colormap)
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    # Create a legend with cluster labels
    legend_labels = [f'Cluster {i}' for i in range(num_clusters)]
    legend = plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
    plt.colorbar(label='Cluster')
    plt.show()
    return

# Itemsets
def run_apriori(data, min_support = 0.34, min_confidence = 0.5):
    #TODO: Check if matrix is binary
    itemsets, rules = apriori(data, min_support=min_support, min_confidence=min_confidence, verbosity=2)
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
    parser.add_argument('--data', type=str, choices=['player_bool', 'player_sum'], default='player_bool',
                        help='Column to use as data (default: player_bool)')
    return parser.parse_args()


# Main
if __name__ == "__main__":
    args = parse_arguments()
    file_path = args.file_path
    original_data, original_item_names = read_data_matrix(file_path)
    label_column = None
    data_column = None
    if args.label == 'date':
        label_column = original_data[:, 0]  # Use the date column as labels
    elif args.label == 'outcome':
        label_column = original_data[:, 1]  # Use the outcome column as labels
    elif args.label == 'competition':
        label_column = original_data[:, 2]  # Use the competition column as labels
    if args.data == 'player_bool':
        data_columns, row_labels = derive_voted(original_data[:,3:], original_item_names[3:])
    elif args.data == 'player_sum':
        data_columns, row_labels = derive_sum_votes(original_data[:,3:], original_item_names[3:])
    #Run Apriori
    transactions = to_transactions(data_columns, row_labels)
    print("\nTransaction List:")
    print(transactions)
    run_apriori(transactions)
    #Run K-means
    run_kmeans(data_columns, 3, label_column)