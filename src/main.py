import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from efficient_apriori import apriori
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Function to read the original data matrix
def read_data_matrix(file_path):
    try:
        data = pd.read_csv(file_path, delimiter=',', index_col=0)
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
def run_kmeans(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    clusters = kmeans.fit_predict(data)

    # Reduce dimensionality for plot
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    # Visualize the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.show()
    print(clusters)
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


# Main
if __name__ == "__main__":
    file_path = "../data/data_raw.csv"
    original_data, original_item_names = read_data_matrix(file_path)
    
    #Compute derived matrix
    voted_data, voted_labels = derive_voted(original_data, original_item_names)
    sum_votes_data, voted_labels = derive_sum_votes(original_data, original_item_names)

    transactions = to_transactions(voted_data, voted_labels)
    print("\nTransaction List:")
    print(transactions)
    #Run Apriori
    run_apriori(transactions)

    #Run K-means
    run_kmeans(sum_votes_data, 3)