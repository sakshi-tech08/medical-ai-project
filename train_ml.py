

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("Data/disease.csv")
print(data.columns)
# Split features and target
X = data.drop("Disease", axis=1)
y = data["Disease"]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = RandomForestClassifier()
model.fit(X, y_encoded)

# Create models folder if not exists
import os
if not os.path.exists("models"):
    os.makedirs("models")

# Save model
pickle.dump(model, open("models/rf_model.pkl", "wb"))
pickle.dump(le, open("models/label_encoder.pkl", "wb"))

print("Model Saved Successfully!")