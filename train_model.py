import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Simple training script to (re)train the ensemble model used by the app.
# Usage: python train_model.py

DATA_PATH = 'dataset.csv'
LABEL_COL = 'ClassLabel'

print('Loading dataset...')
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[LABEL_COL])

X = df.drop(columns=[LABEL_COL, 'URL'], errors='ignore')
y = df[LABEL_COL]

# Basic preprocessing: impute, scale
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Build a small ensemble
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000)

voting = VotingClassifier(estimators=[('rf', rf), ('lr', lr)], voting='soft')

print('Training ensemble...')
voting.fit(X_scaled, y_encoded)

# Save artifacts: scaler, imputer, label encoder, and model in a dict
model_bundle = {
    'imputer': imputer,
    'scaler': scaler,
    'model': voting
}

with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(voting, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print('Saved ensemble_model.pkl and label_encoder.pkl')

# Quick eval
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
pred = voting.predict(X_test)
print('Accuracy:', accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=le.classes_))
