import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("insta_train.csv")

# Drop unnecessary columns
df = df.drop(["profile pic", "name==username", "description length", "external URL"], axis=1)

# Features & target selection
X = df.drop("fake", axis=1)
Y = df["fake"]

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

# Save model and scaler
joblib.dump(rf_model, "random_forest.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model & Scaler saved!")