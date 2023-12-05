import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create DataFrame by reading the dataset_1.csv
df = pd.read_csv('../dataset/dataset_1.csv')

# Initialize the LabelEncoder
le = LabelEncoder()

# Normalize all the labels
df['type'] = le.fit_transform(df['type'])
df['nameOrig'] = le.fit_transform(df['nameOrig'])
df['nameDest'] = le.fit_transform(df['nameDest'])

# Set X and y data
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Divide all the datas as a set of training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créez un modèle de réseau de neurones
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilez le modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînez le modèle
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Faites des prédictions sur l'ensemble de test
predictions = (model.predict(X_test) > 0.5).astype("int32")

# Evaluate the model's performance by calculating the precision and the error's rate
accuracy = np.sum(predictions == y_test) / len(y_test)
error_rate = 1 - accuracy

# Display in terminal all the result
print("Précision du modèle : {:.2f}%".format(accuracy * 100))
print("Taux d'erreur du modèle : {:.2f}%".format(error_rate * 100))

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Matrice de confusion :\n", conf_matrix)

# Create a heatmap with seaborn
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Classe négative', 'Classe positive'],
            yticklabels=['Classe négative', 'Classe positive'])

plt.title('Matrice de Confusion')
plt.xlabel('Prédictions')
plt.ylabel('Vraies Valeurs')

# Save the confusion_matrix file as png
plt.savefig('confusion_matrix.png')

# Show the confusion_matrix
plt.show()