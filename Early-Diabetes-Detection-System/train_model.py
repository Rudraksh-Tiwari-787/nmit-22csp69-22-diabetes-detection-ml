import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
dataset = pd.read_csv("diabetes.csv")

# Features and target
X = dataset.drop("Outcome", axis=1)
y = dataset["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "diabetes_model.pkl")

print("Model Trained and Saved Successfully")