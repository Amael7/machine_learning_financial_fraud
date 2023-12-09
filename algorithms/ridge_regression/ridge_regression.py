import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
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

# Create a Ridge Regression model
ridge_model = Ridge(alpha=1.0)  # You can adjust the alpha parameter for regularization

# Train the model with the training set
ridge_model.fit(X_train, y_train)

# Make predictions on the testing dataset
predictions = ridge_model.predict(X_test)

# Evaluate the model's performance by calculating Mean Squared Error (MSE) and R-squared
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Display results in the terminal
print("Mean Squared Error: {:.2f}".format(mse))
print("R-squared: {:.2f}".format(r2))

# Display a scatter plot of predicted vs. actual values
plt.scatter(y_test, predictions, alpha=0.5)
plt.title('Ridge Regression - Predicted vs. Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Save the confusion_matrix file as png
plt.savefig('./confusion_matrix.png')

plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
