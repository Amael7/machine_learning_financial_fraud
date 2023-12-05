import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create DataFrame by reading the dataset_1.csv
df = pd.read_csv('../../dataset/cleaned_dataset.csv')

# Set X and y data
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Divide all the datas as a set of training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a classificator KNN with a given neighbor's number
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the model with the previous training set of data
knn_classifier.fit(X_train, y_train)

# Make prediction on training dataset
predictions = knn_classifier.predict(X_test)

# Evaluate the model's performance by calculating the precision
accuracy = accuracy_score(y_test, predictions)

# Set the error rate
error_rate = 1 - accuracy

# Display in terminal all the result
print("Model Accuracy : {:.2f}%".format(accuracy * 100))
print("Model Error Rate : {:.2f}%".format(error_rate * 100))

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix :\n", conf_matrix)

# Create a heatmap with seaborn
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Negative Class', 'Positive Class'],
            yticklabels=['Negative Class', 'Positive Class'])

plt.title('Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('Real values')

# Save the confusion_matrix file as png
plt.savefig('./confusion_matrix.png')

# Show the confusion_matrix
plt.show()
