import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Save the start time of the file execution
start_time = time.time()

# Create DataFrame by reading the original_dataset.csv
df = pd.read_csv('../../dataset/synthetic_dataset.csv')

# Set X and y data
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Divide all the data as a set of training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Lasso Regression model
lasso_model = Lasso(alpha=1.0)  # You can adjust the alpha parameter for regularization

# Train the model with the training set
lasso_model.fit(X_train, y_train)

# Make predictions on the testing dataset (considering a threshold of 0.5)
predictions_proba = lasso_model.predict(X_test)
predictions = (predictions_proba >= 0.5).astype(int)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

# Display results in the terminal
print("Model Accuracy : {:.2f}%".format(accuracy * 100))
print("Confusion Matrix :\n", conf_matrix)
print("Classification Report :\n", classification_report(y_test, predictions))

# Display a heatmap with seaborn
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

print("--- %s seconds ---" % (time.time() - start_time))
