import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import time

# Save the start time of the file execution
start_time = time.time()

# Create DataFrame by reading the original_dataset.csv
df = pd.read_csv('../../dataset/synthetic_dataset.csv')

# Assume the dataset is in a transactional format where each row represents a transaction
transactions = df.applymap(str).values.tolist()

# Convert the transactions to a one-hot encoded format
ohe_df = pd.get_dummies(df, drop_first=True)

# Perform Apriori algorithm
frequent_itemsets = apriori(ohe_df, min_support=0.05, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display frequent itemsets and association rules (optional)
print("Frequent Itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", rules)

print("--- %s seconds ---" % (time.time() - start_time))
