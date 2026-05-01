import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# -------------------------------
# STEP 1: LOAD DATASET
# -------------------------------
df = pd.read_csv("data/creditcard.csv")
# -------------------------------
# REDUCE DATA SIZE (IMPORTANT)
# -------------------------------
df = df.sample(n=50000, random_state=42)

print("Reduced dataset size for faster processing ✅")

print("Dataset Loaded Successfully ✅")
print("Shape:", df.shape)

# -------------------------------
# STEP 2: PREVIEW DATA
# -------------------------------
print("\nFirst 5 rows:")
print(df.head())

# -------------------------------
# STEP 3: BASIC INFO
# -------------------------------
print("\nTotal Columns:", len(df.columns))
print("First 10 Columns:", list(df.columns[:10]))

# -------------------------------
# STEP 4: CLASS DISTRIBUTION
# -------------------------------
print("\nClass Distribution:")
print(df['Class'].value_counts())
# -------------------------------
# STEP 7: FEATURES & TARGET
# -------------------------------
X = df.drop('Class', axis=1)
y = df['Class']

print("\nFeatures and Target created ✅")
print("X shape:", X.shape)
print("y shape:", y.shape)

# -------------------------------
# STEP 8: SCALING
# -------------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

print("\nScaling Done ✅")

# -------------------------------
# STEP 9: TRAIN-TEST SPLIT
# -------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split Done ✅")
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# -------------------------------
# STEP 10: APPLY SMOTE
# -------------------------------
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nSMOTE Applied ✅")

print("\nBefore SMOTE:")
print(y_train.value_counts().to_string(index=True))

print("\nAfter SMOTE:")
print(y_train_smote.value_counts().to_string(index=True))
# -------------------------------
# STEP 11: MODEL TRAINING
# -------------------------------
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_smote, y_train_smote)

print("\nModel Trained Successfully ✅")

# -------------------------------
# STEP 12: PREDICTIONS
# -------------------------------
y_pred = model.predict(X_test)

print("\nPredictions Done ✅")

# -------------------------------
# STEP 13: EVALUATION
# -------------------------------
from sklearn.metrics import classification_report, confusion_matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n--- Logistic Regression Results ---")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# RANDOM FOREST MODEL
# -------------------------------
print("\n🔥 RUNNING RANDOM FOREST 🔥")

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)

print("Training RF...")
rf_model.fit(X_train_smote, y_train_smote)

print("Predicting RF...")
y_pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest Results ---")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# -------------------------------
# STEP 5: VISUALIZATION
# -------------------------------
os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(8,5))
sns.countplot(x='Class', data=df)

plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("outputs/class_distribution.png")
plt.show()

# -------------------------------
# CONFUSION MATRIX VISUALIZATION
# -------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("outputs/confusion_matrix_rf.png")
plt.show()

import joblib
joblib.dump(rf_model, "models/fraud_model.pkl")
print("Model saved ✅")