# Battery Aging Classification Script
# Author: Usama Yasir Khan

"""
This script is designed for battery aging classification using machine learning techniques. 
It involves feature engineering, data preprocessing, model training, and evaluation to predict 
the aging stages of a lithium-ion battery (Healthy, Moderate Aging, Aged) based on parameters 
such as voltage, current, temperature, and capacity. A Random Forest Classifier is used for 
classification, with the aim to provide insights into battery health and potential degradation.
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
data = pd.read_csv('battery_aging_data.csv', parse_dates=['Time Stamp'], index_col='Time Stamp')

# Feature engineering
# Calculate rolling averages and change in capacity
data['Voltage_Avg'] = data['Voltage [V]'].rolling(window=5, min_periods=1).mean()
data['Current_Avg'] = data['Current [A]'].rolling(window=5, min_periods=1).mean()
data['Temperature_Avg'] = data['Temperature [C]'].rolling(window=5, min_periods=1).mean()
data['Delta_Capacity'] = data['Capacity [Ah]'].diff().fillna(0)

# Create aging labels based on capacity thresholds
initial_capacity = data['Capacity [Ah]'].iloc[0]
data['Aging_Label'] = pd.cut(data['Capacity [Ah]'],
                              bins=[-np.inf, 0.7 * initial_capacity, 0.85 * initial_capacity, np.inf],
                              labels=['Aged', 'Moderate Aging', 'Healthy'])

# Select features for the model
features = data[['Voltage [V]', 'Current [A]', 'Temperature [C]', 'Capacity [Ah]',
                 'Voltage_Avg', 'Current_Avg', 'Temperature_Avg', 'Delta_Capacity']]
labels = data['Aging_Label']

# Encode target labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the data into training and testing sets (time-based split)
split_index = int(len(features) * 0.8)
X_train, X_test = features[:split_index], features[split_index:]
y_train, y_test = labels_encoded[:split_index], labels_encoded[split_index:]

# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
display.plot(cmap='Blues')
plt.title('Confusion Matrix for Battery Aging Classification')
plt.show()

# Plot voltage, current, and aging stage over time
plt.figure(figsize=(14, 12))

# Plot voltage over time
plt.subplot(3, 1, 1)
plt.plot(data.index, data['Voltage [V]'], color='orange')
plt.title('Voltage Over Time')
plt.ylabel('Voltage [V]')

# Plot current over time
plt.subplot(3, 1, 2)
plt.plot(data.index, data['Current [A]'], color='green')
plt.title('Current Over Time')
plt.ylabel('Current [A]')

# Plot aging stage over time
plt.subplot(3, 1, 3)
plt.plot(data.index, labels_encoded, label='Actual Aging Stage', linestyle='-', color='blue')
plt.plot(data.index[split_index:], y_pred, label='Predicted Aging Stage', linestyle='--', color='red')
plt.title('Aging Stage Over Time')
plt.xlabel('Time')
plt.ylabel('Aging Stage')
plt.legend()

plt.tight_layout()
plt.show()

# Feature Importance
importances = model.feature_importances_
feature_names = features.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='teal')
plt.title('Feature Importance in Battery Aging Classification')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Save the model and scaler for future use
import joblib
joblib.dump(model, 'battery_aging_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")
