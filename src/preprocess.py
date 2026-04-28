import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    # Drop useless column
    df = df.drop(['Point of Contact'], axis=1)

    # Fix Floor column
    df['Floor'] = df['Floor'].str.extract(r'(\d+)')
    df['Floor'] = df['Floor'].fillna(0).astype(int)

    # Remove outliers (VERY IMPORTANT)
    df = df[(df['Rent'] < 200000) & (df['Rent'] > 1000)]

    return df

def preprocess(df):
    df = clean_data(df)

    # Target
    y = df['Rent']

    # Features
    X = df.drop('Rent', axis=1)

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    return X, y