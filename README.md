# FCB Player Ratings
This AI application for team analysis which runs on toy data. Real data would include real time tracking of player statistics. This repo uses FCB player ratings of the 2023/2024 season, gathered from RAC1 radio streams.
The toy data includes a csv file with the ratings per game, together with the outcomes, the date and competition (data_raw.csv), and another csv with the player's positions (players_raw). That's all the input data.

# Options
Data can be preprocessed in a few ways:
* original. Not recommended for any of the analysis due to the disaggregated nature (each player represented by 3 columns).
* player_bool. Reduces the data to, was the player voted at least once this game? Appropriate for itemsets.
* player_sum. Sums the total points per player. Appropriate for most methods (not itemsets)
* position_sum. Sums the total points per position (GK, DF, MF, FW). Appropriate for most methods (not itemsets)
* sum_combined. Concatenates player_sum and position_sum features. For comparison purposes.

Labels can be the following:
* date. Day and month of the game (e.g., 28-09)
* outcome. (V)ictoria, (E)mpat, (D)errota
* comptetition. (0) La liga, (1) Champions League  

Several features are currently implemented, which can be applied to players, positions, or both:
* Ranking by total points (--analysis ranking)
* Produce an animated gif of the accumulation of points per player (--analysis accumulated_gif)
* Decision trees. Using as target (--label), builds a dendogram (--analysis dt)
* Hierarchical clustering. Using as x-axis (--label) builds an agglomerative tree (--analysis hc)
* Kmeans. Using (--label) to paint a PCA plot, runs this algorithm to assign clusters (--analysis kmeans)
* Itemsets. Runs the apriori algorithm to find frequent itemsets. First transforms data into binary transactions (--analysis itemsets)


# Examples
Producing the overall ranking:
`python main.py --label date --data sum_combined ../data/data_raw.csv --analysis ranking`

To create the accumulated plot by position:
`python main.py --label date --data position_sum ../data/data_raw.csv --analysis accumulated_gif`

and by player:
`python main.py --label date --data player_sum ../data/data_raw.csv --analysis accumulated_gif`

Frequent itemsets and association rules by player:
`python main.py --label date --data player_bool ../data/data_raw.csv --analysis itemsets`

# License
This software is released under the MIT License. See LICENSE file.
