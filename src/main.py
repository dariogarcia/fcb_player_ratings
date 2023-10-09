import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from efficient_apriori import apriori
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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

def to_transactions(data, item_names):
    #Asumes matrix is binary
    if not np.all(np.logical_or(data == 0, data == 1)):
        warnings.warn("The data matrix is not binary. Non-binary values will be treated as 1's.")
    # transforms a data matrix into a list of transactions
    transactions = []
    for row in data:
        transaction = [item_names[i] for i, item in enumerate(row) if item >= 1]
        transactions.append(tuple(transaction))
    return transactions

# Clustering
def run_kmeans(data, num_clusters, labels=None):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=20)
    clusters = kmeans.fit_predict(data)
    print(num_clusters,"-Clusters Found:")
    for i in range(num_clusters):
        cluster_indices = np.where(clusters == i)[0]
        print(f"Cluster {i}: {len(cluster_indices)} data points")

    # Define a colormap with a fixed number of colors
    colormap = plt.cm.get_cmap('Dark2_r', num_clusters)
    # Reduce dimensionality for plot
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    
    # Visualize the clusters with labels
    plt.figure(figsize=(8, 6))
    
    symbols = ['o', '*', 's', '^', 'D', 'v', 'p', 'H', '+']
    unique_clusters = np.unique(clusters)
    unique_labels = np.unique(labels)
    print('labels:',unique_labels)
    for i, cluster in enumerate(unique_clusters):
        cluster_indices = np.where(clusters == cluster)
        label = f'Cluster {cluster}'
        symbol = symbols[i % len(symbols)]
        
        for j, label_value in enumerate(unique_labels):
            label_indices = np.where(labels == label_value)
            intersect_indices = np.intersect1d(cluster_indices, label_indices)
            
            if len(intersect_indices) > 0:
                label = f'Cluster {cluster}, Label {label_value}'
                symbol = symbols[j % len(symbols)]
                scatter =plt.scatter(reduced_data[intersect_indices, 0], reduced_data[intersect_indices, 1],
                            label=label, c=colormap(i % num_clusters), marker=symbol)
    
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    # Create a legend with cluster and label combinations
    legend_labels = [f'Cluster {i}' for i in unique_clusters]
    for j, label_value in enumerate(unique_labels):
        legend_labels.extend([f'Cluster {i}, Label {label_value}' for i in unique_clusters])
    legend = plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
    plt.colorbar(label='Cluster')
    plt.show()
    return

def run_decision_tree(data, labels, feature_names, max_depth=None):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(data, labels)
    tree_text = export_text(clf, feature_names=feature_names)
    print("\nDecision Tree:")
    print(tree_text)
    # Feature importances
  
    # Make predictions
    predictions = clf.predict(data)
    # Calculate performance metrics
    accuracy = accuracy_score(labels, predictions)
    confusion = confusion_matrix(labels, predictions)
    classification_rep = classification_report(labels, predictions)

    # Print performance metrics
    print("\nPerformance Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion}")
    print(f"Classification Report:\n{classification_rep}")

    return clf  # Return the trained Decision Tree classifier


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
    parser.add_argument('--data', type=str, choices=['original','player_bool', 'player_sum'], default='player_bool',
                        help='Column to use as data (default: player_bool)')
    return parser.parse_args()


# Main
if __name__ == "__main__":
    args = parse_arguments()
    file_path = args.file_path
    original_data, original_item_names = read_data_matrix(file_path)
    label_column, data_column = None, None
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
    #Run Apriori
    #run_apriori(data_columns, row_labels)
    #Run K-means
    #run_kmeans(data_columns, 3, label_column)
    #Run Decision Tree
    run_decision_tree(data_columns, label_column, row_labels, max_depth=5)