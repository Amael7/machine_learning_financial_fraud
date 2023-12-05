import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Save the start time of the file execution
start_time = time.time()

# Create DataFrame by reading the dataset_1.csv
df = pd.read_csv('../../dataset/cleaned_dataset.csv')

# Set X data (no y since Hierarchical Clustering is unsupervised)
X = df.drop('isFraud', axis=1)

# Perform Hierarchical Clustering
hierarchical_clustering = AgglomerativeClustering(n_clusters=2)  # Specify the number of clusters
cluster_assignments = hierarchical_clustering.fit_predict(X)

# Display cluster assignments (optional)
print("Cluster Assignments:\n", cluster_assignments)

# Create a heatmap with seaborn to visualize the clusters
sns.heatmap(pd.DataFrame(cluster_assignments, columns=['Cluster']), cmap="Blues", cbar=False,
            xticklabels=['Cluster'], yticklabels=['Sample'])

plt.title('Hierarchical Clustering - Cluster Assignments')
plt.xlabel('Cluster')
plt.ylabel('Sample')

# Save the confusion matrix plot as png
plt.savefig('./confusion_matrix.png')

# Show the clustering plot
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
