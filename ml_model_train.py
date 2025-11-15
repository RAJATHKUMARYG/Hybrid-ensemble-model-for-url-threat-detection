import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load test dataset
test_df = pd.read_csv('dataset.csv')
LABEL_COL = 'ClassLabel'  # Update with your label column name

# Drop rows with missing labels
test_df = test_df.dropna(subset=[LABEL_COL])

# Separate features and labels
X_test = test_df.drop(columns=[LABEL_COL, 'URL'])  # Drop non-numeric 'URL'
y_test = test_df[LABEL_COL]

# Replace infinite values with NaN
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Impute missing values (using mean)
imputer = SimpleImputer(strategy='mean')
X_test_imputed = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns)

# Load trained label encoder and model
with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

with open('ensemble_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Encode test labels to match training
y_test_encoded = le.transform(y_test)

# Predict on test data
pred_encoded = model.predict(X_test_imputed)

# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, pred_encoded)
print('Accuracy on lexical features:', accuracy)
