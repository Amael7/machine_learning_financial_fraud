import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

def clean_dataset(df):
    # Delete missing values
    df = df.dropna()

    # Delete duplicates
    df = df.drop_duplicates()

    # Convert data types into numeric type
    # Initialize the LabelEncoder
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])
    df['nameOrig'] = le.fit_transform(df['nameOrig'])
    df['nameDest'] = le.fit_transform(df['nameDest'])

    # Normalize numeric data
    scaler = StandardScaler()
    numeric_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Data balancing to obtain data that is as fraudulent as it is clean in a balanced way
    x_resampled, y_resampled = SMOTE().fit_resample(df.drop(['isFraud', 'isFlaggedFraud'], axis=1), df['isFraud'])
    df = pd.concat([x_resampled, y_resampled], axis=1)

    return df

def create_synthetic_dataset(df):
    df = df.sample(frac=0.1)

    return df

def create_dataset_with_missing_data(df):
    df = df.sample(frac=0.1)

    # Replace some random value in the dataset with Nan value
    df['amount'] = np.where(np.random.rand(len(df)) < 0.3, np.nan, df['amount'])
    df['oldbalanceOrg'] = np.where(np.random.rand(len(df)) < 0.3, np.nan, df['oldbalanceOrg'])
    df['newbalanceOrig'] = np.where(np.random.rand(len(df)) < 0.3, np.nan, df['newbalanceOrig'])
    df['oldbalanceDest'] = np.where(np.random.rand(len(df)) < 0.3, np.nan, df['oldbalanceDest'])
    df['newbalanceDest'] = np.where(np.random.rand(len(df)) < 0.3, np.nan, df['newbalanceDest'])

    return df


if __name__ == "__main__":
    # Load Dataset
    data = pd.read_csv("dataset/original_dataset.csv")

    # 0.Clean_dataset
    cleaned_data = clean_dataset(data)

    # Make 3 types of dataset (large, synthetic, missing_values)
    # 1.Large
    large_data = cleaned_data

    # 2.Synthetic
    synthetic_data = create_synthetic_dataset(cleaned_data)

    # # 3.Missing_values
    missing_data = create_dataset_with_missing_data(cleaned_data)

    # # Save them on new CSV files
    cleaned_data.to_csv("dataset/cleaned_dataset.csv", index=False)
    large_data.to_csv("dataset/large_dataset.csv", index=False)
    synthetic_data.to_csv("dataset/synthetic_dataset.csv", index=False)
    missing_data.to_csv("dataset/dataset_with_missing_values.csv", index=False)
