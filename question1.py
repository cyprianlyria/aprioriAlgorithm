import pandas as pd
from itertools import combinations
from collections import defaultdict
import time

def read_transactions(file_name):
    try:
        # Assuming "trans.txt" is in the same directory as the script
        df = pd.read_csv(file_name, header=None, delimiter=' ')
        transactions = df.values.tolist()
        return transactions
    except FileNotFoundError:
        print(f"File '{file_name}' not found. Please make sure the file is in the correct directory.")
        return None

def generate_candidates(itemset, k):
    return set([i.union(j) for i in itemset for j in itemset if len(i.union(j)) == k])

def prune_candidates(itemset, prev_candidates):
    return set([i for i in itemset if all(subset in prev_candidates for subset in combinations(i, len(i)-1))])

def apriori(data, min_support):
    transactions = [set(transaction) for transaction in data]
    itemset = set([frozenset([item]) for transaction in transactions for item in transaction])
    frequent_itemsets = dict()

    k = 1
    while itemset:
        candidates = generate_candidates(itemset, k)
        candidates = prune_candidates(candidates, itemset)

        item_counts = defaultdict(int)
        for transaction in transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    item_counts[candidate] += 1

        frequent_itemsets[k] = {item: count for item, count in item_counts.items() if count >= min_support}
        itemset = set(frequent_itemsets[k].keys())
        k += 1

    return frequent_itemsets

def run_apriori_and_measure_time(data, min_support):
    start_time = time.time()
    frequent_itemsets = apriori(data, min_support)
    elapsed_time = time.time() - start_time
    return frequent_itemsets, elapsed_time

# Assuming "trans.txt" is in the same directory as the script
file_name = "trans.txt"
transactions = read_transactions(file_name)

if transactions is not None:
    # Varying minimum support values
    min_support_values = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]

    for min_support in min_support_values:
        frequent_itemsets, elapsed_time = run_apriori_and_measure_time(transactions, min_support)

        print(f"\nMinimum Support: {min_support}")
        for k, itemsets in frequent_itemsets.items():
            print(f"Number of frequent {k}-itemsets: {len(itemsets)}")
            print(f"Frequent {k}-itemsets:")
            for itemset, count in itemsets.items():
                print(f"{set(itemset)} - Support: {count}")
        print(f"Elapsed Time: {elapsed_time} seconds")
