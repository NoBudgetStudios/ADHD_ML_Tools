# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
dataset = pd.read_csv('dataset.csv')

# Splitting features and target
X = dataset.drop(['user_id', 'has_adhd'], axis=1)  # Features
y = dataset['has_adhd']  # Target

# Apply SMOTE to balance the class distribution
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split on the balanced data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Training Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions
y_pred = rf_model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)

# Class Imbalance Metrics
class_counts = y_resampled.value_counts()
class_imbalance_ratio = class_counts[1] / class_counts[0]
print("Class Imbalance Ratio:", class_imbalance_ratio)

# Saving the model
joblib.dump(rf_model, 'adhd_random_forest_model.pkl')
