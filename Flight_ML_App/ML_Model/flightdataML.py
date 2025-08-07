import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv(os.path.join("ML_Model", "airlines_flights_data.csv"))

# Example: Let's predict if price is above median (binary classification)
median_price = df['price'].median()
y = (df['price'] > median_price).astype(int)  # 1 if above median, else 0
x = df.drop(['price'], axis=1)

# Convert categorical columns to numeric
x = pd.get_dummies(x, drop_first=True)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Save model
pickle.dump(model, open(os.path.join("ML_Model", "flight_price_logistic_model.pkl"), "wb"))

# Predict and print accuracy
predictions = model.predict(x_test)
print("Accuracy:", np.mean(predictions == y_test))