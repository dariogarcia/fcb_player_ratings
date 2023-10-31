import argparse
import numpy as np
from time_gif import create_accumulated_votes_animation
from mydt import run_decision_tree, run_decision_tree_cv
from myclustering import run_hierarchical_clustering, run_kmeans
from data_loaders import read_data_matrix, derive_voted, derive_sum_votes, derive_pos_sum_votes, player_rankings
from myitemsets import run_apriori

def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform data analysis on FCB player ratings.')
    parser.add_argument('file_path', type=str, help='Path to the input CSV file')
    parser.add_argument('--label', type=str, choices=['date', 'outcome', 'competition'], default=None,
                        help='Column to use as label (default: None)')
    parser.add_argument('--data', type=str, choices=['original','player_bool', 'player_sum', 'position_sum', 'sum_combined'], default='player_bool',
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
    if args.data == 'sum_combined':
        data_columns_1, row_labels_1 = derive_pos_sum_votes(original_data[:,3:], original_item_names[3:], '../data/players_raw.csv')
        data_columns_2, row_labels_2 = derive_sum_votes(original_data[:,3:], original_item_names[3:])
        data_columns = np.concatenate((data_columns_1, data_columns_2), axis=1)
        row_labels = np.concatenate((row_labels_1, row_labels_2))
    
    #Run Apriori
    run_apriori(data_columns, row_labels)
    
    #Run K-means
    run_kmeans(data_columns, 3, label_column)

    #Run Hierarchical Clustering
    run_hierarchical_clustering(data_columns, label_column)

    #Run Decision Tree
    run_decision_tree(data_columns, label_column, row_labels, max_depth=5)
    #run_decision_tree_cv(data_columns, label_column, row_labels, max_depth=5)

    #Create accumulated animation
    if  args.data != 'original':
        create_accumulated_votes_animation(data_columns, row_labels)
    
    #Pring player rankings
    data_columns, row_labels = derive_sum_votes(original_data[:,3:], original_item_names[3:])
    player_rankings(data_columns, row_labels)