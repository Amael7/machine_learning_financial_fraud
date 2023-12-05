import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv('../../dataset/cleaned_dataset.csv')

# Set X and y data
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a LightGBM Regressor
lgbm_regressor = LGBMRegressor(n_estimators=100, random_state=42)

# Train the model with the training data
lgbm_regressor.fit(X_train, y_train)

# Make predictions on the testing dataset
predictions = lgbm_regressor.predict(X_test)

# Evaluate the model's performance by calculating Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)

# Display results in the terminal
print("Mean Squared Error: {:.2f}".format(mse))

# Display the scatter plot of predicted vs. actual values
plt.scatter(y_test, predictions, alpha=0.5)
plt.title('LightGBM Regression - Predicted vs. Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Save the confusion_matrix file as png
plt.savefig('./light_boosting_regression.png')

plt.show()
