import pandas as pd
from collections import defaultdict
import time

class HashTreeNode:
    def __init__(self, item, count, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}

def read_transactions(file_name):
    try:
        df = pd.read_csv(file_name, header=None, delimiter=' ')
        transactions = df.values.tolist()
        return transactions
    except FileNotFoundError:
        print(f"File '{file_name}' not found. Please make sure the file is in the correct directory.")
        return None

def generate_candidates(itemset, k):
    return set([i.union(j) for i in itemset for j in itemset if len(i.union(j)) == k])

def generate_hash_tree(data, min_support):
    transactions = [set(transaction) for transaction in data]
    item_counts = defaultdict(int)

    # Count items in the dataset
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] += 1

    # Filter items based on min_support
    frequent_items = {item: count for item, count in item_counts.items() if count >= min_support}
    sorted_frequent_items = sorted(frequent_items.items(), key=lambda x: x[1], reverse=True)

    # Construct hash tree
    root = HashTreeNode(None, 0)
    for transaction in transactions:
        transaction = [item for item in transaction if frozenset([item]) in frequent_items]
        transaction.sort(key=lambda x: frequent_items[frozenset([x])], reverse=True)
        insert_transaction(root, transaction)

    return root, sorted_frequent_items

def insert_transaction(node, transaction):
    if not transaction:
        return

    item = transaction[0]
    if item not in node.children:
        child = HashTreeNode(item, 1, parent=node)
        node.children[item] = child
        insert_transaction(child, transaction[1:])
    else:
        node.children[item].count += 1
        insert_transaction(node.children[item], transaction[1:])

def mine_frequent_patterns(node, min_support, current_prefix, frequent_patterns):
    if node.count >= min_support:
        frequent_patterns[frozenset(current_prefix)] = node.count

    for child_item, child_node in node.children.items():
        new_prefix = current_prefix + [child_item]
        mine_frequent_patterns(child_node, min_support, new_prefix, frequent_patterns)

def apriori_optimized(data, min_support):
    root, frequent_items = generate_hash_tree(data, min_support)
    frequent_patterns = dict()

    for frequent_item, count in frequent_items:
        mine_frequent_patterns(root.children[frequent_item], min_support, [frequent_item], frequent_patterns)

    return frequent_patterns

def run_apriori_and_measure_time(data, min_support, optimized=False):
    start_time = time.time()
    
    if optimized:
        frequent_itemsets = apriori_optimized(data, min_support)
    else:
        frequent_itemsets = {}  # Placeholder for the case where apriori function is not defined
    
    elapsed_time = time.time() - start_time
    return frequent_itemsets, elapsed_time

file_name = "trans.txt"
transactions = read_transactions(file_name)

if transactions is not None:
    min_support_values = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]

    for min_support in min_support_values:
        print(f"\nMinimum Support: {min_support}")

        # Without optimization
        frequent_itemsets, elapsed_time = run_apriori_and_measure_time(transactions, min_support)
        print("Without optimization:")
        print("Frequent Itemsets:", frequent_itemsets)
        print(f"Elapsed Time: {elapsed_time} seconds")

        # With optimization
        frequent_itemsets_optimized, elapsed_time_optimized = run_apriori_and_measure_time(transactions, min_support, optimized=True)
        print("\nWith optimization:")
        print("Frequent Itemsets:", frequent_itemsets_optimized)
        print(f"Elapsed Time: {elapsed_time_optimized} seconds")
