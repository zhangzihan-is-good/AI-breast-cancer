import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Read the training and testing sets
train_file_path = 'Dataset_muti/training_set.xlsx'
test_file_path = 'Dataset_muti/testing_set.xlsx'
train_data = pd.read_excel(train_file_path)
test_data = pd.read_excel(test_file_path)

# Map the text in the Benign or Malignant column to numbers
label_encoder = LabelEncoder()
train_data['Benign or Malignant'] = label_encoder.fit_transform(train_data['Benign or Malignant'])
test_data['Benign or Malignant'] = label_encoder.transform(test_data['Benign or Malignant'])

# Convert other categorical data to numerical
for column in ['Pain', 'Texture', 'Skin Changes', 'Nipple Discharge']:
    train_data[column] = label_encoder.fit_transform(train_data[column])
    test_data[column] = label_encoder.transform(test_data[column])

# Separate features and labels
X_train = train_data[['Pain', 'Texture', 'Skin Changes', 'Nipple Discharge']]
y_train = train_data['Benign or Malignant']
X_test = test_data[['Pain', 'Texture', 'Skin Changes', 'Nipple Discharge']]
y_test = test_data['Benign or Malignant']

# Initialize the XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth = 4, learning_rate = 0.01, n_estimators = 200, subsample = 0.8, colsample_bytree = 0.2, gamma = 0)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict using the model
y_pred = xgb_model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print performance metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')
