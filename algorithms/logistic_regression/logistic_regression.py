import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create DataFrame by reading the dataset_1.csv
df = pd.read_csv('../../dataset/cleaned_dataset.csv')

# Set X and y data
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Divide all the data as a set of training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
logistic_reg_model = LogisticRegression()

# Train the model with the training set
logistic_reg_model.fit(X_train, y_train)

# Make predictions on the testing dataset
predictions = logistic_reg_model.predict(X_test)

# Evaluate the model's performance by calculating accuracy and displaying the confusion matrix
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

# Display results in the terminal
print("Model Accuracy: {:.2f}%".format(accuracy * 100))

# Display the confusion matrix
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
