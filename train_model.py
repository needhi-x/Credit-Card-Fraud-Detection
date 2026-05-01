import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# ALL FEATURES (IMPORTANT)
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale Amount
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# Save scaler (IMPORTANT for API consistency)
joblib.dump(scaler, "models/scaler.pkl")

# Balance dataset
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/fraud_model.pkl")

print("✅ Model trained successfully")