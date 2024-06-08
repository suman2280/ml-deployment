# src/scripts/train.py
import numpy as np
import pandas as pd
from models.model.py import SimpleModel

# Generate some example data
X = np.array([[i] for i in range(100)])
y = np.array([2*i + 1 for i in range(100)])

model = SimpleModel()
model.train(X, y)

# Save the trained model
model.save_model('model/model.joblib')

print("Model trained and saved successfully.")
