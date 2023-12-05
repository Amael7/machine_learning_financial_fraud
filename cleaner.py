import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    # Delete missing values
    df = df.dropna()

    # Delete duplicates
    df = df.drop_duplicates()

    # Convert data types if needed
    # Initialize the LabelEncoder
    le = LabelEncoder()

    df['type'] = le.fit_transform(df['type'])
    df['nameOrig'] = le.fit_transform(df['nameOrig'])
    df['nameDest'] = le.fit_transform(df['nameDest'])

    return df


if __name__ == "__main__":
    # Load Dataset
    dataset_path = "dataset/dataset_1.csv"
    data = pd.read_csv(dataset_path)

    # Clean data
    cleaned_data = clean_data(data)

    # Save data on new CSV file
    cleaned_data.to_csv("dataset/cleaned_dataset.csv", index=False)
