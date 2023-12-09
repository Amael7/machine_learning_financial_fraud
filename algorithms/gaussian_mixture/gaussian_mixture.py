import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Save the start time of the file execution
start_time = time.time()

# Create DataFrame by reading the original_dataset.csv
df = pd.read_csv('../../dataset/synthetic_dataset.csv')

# Set X data (no y since GMM is unsupervised)
X = df.drop('isFraud', axis=1)

# Perform Gaussian Mixture Models
gmm = GaussianMixture(n_components=2)  # Specify the number of components (clusters)
gmm.fit(X)
cluster_assignments = gmm.predict(X)

# Display cluster assignments (optional)
print("Cluster Assignments:\n", cluster_assignments)

# Create a heatmap with seaborn to visualize the clusters
sns.heatmap(pd.DataFrame(cluster_assignments, columns=['Cluster']), cmap="Blues", cbar=False,
            xticklabels=['Cluster'], yticklabels=['Sample'])

plt.title('Gaussian Mixture Models - Cluster Assignments')
plt.xlabel('Cluster')
plt.ylabel('Sample')

# Save the confusion matrix as png
plt.savefig('./confusion_matrix.png')

# Show the clustering plot
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
