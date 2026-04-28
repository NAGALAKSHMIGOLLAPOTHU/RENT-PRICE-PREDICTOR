import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from preprocess import load_data, preprocess

# Load data
df = load_data('../data/House_Rent_Dataset.csv')

# Preprocess
X, y = preprocess(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model (Better than Linear Regression)
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
with open('../model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save columns
with open('../model/columns.pkl', 'wb') as f:
    pickle.dump(X.columns, f)

print("✅ Model trained successfully!")