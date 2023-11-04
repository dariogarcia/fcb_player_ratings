import numpy as np
import warnings
from efficient_apriori import apriori

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

# Itemsets
def run_apriori(data_columns, row_labels, min_support = 0.25, min_confidence = 0.7):
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