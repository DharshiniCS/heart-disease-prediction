import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
import joblib

# Load trained model and preprocessing objects
model = load_model("cnn_rnn_model.keras")
scaler = joblib.load("scaler.save")

label_classes = np.load("label_encoder_classes.npy", allow_pickle=True)

# Load test data
df = pd.read_csv("csv_dataset/training-dataset_echo.csv")

# Drop unwanted columns and keep only numeric features
X = df.drop(['label', 'stage', 'treatment', 'mortality'], axis=1).select_dtypes(include=[np.number]).values
y = df['label'].values

print("Shape of X:", X.shape)
print("Sample X row:", X[0])

# Recreate the LabelEncoder and transform labels
le = LabelEncoder()
le.classes_ = label_classes
y_encoded = le.transform(y)
y_cat = to_categorical(y_encoded)

# Scale input features
X_scaled = scaler.transform(X)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
print("Input shape before reshape:", X_scaled.shape)

# Make predictions
y_pred_probs = model.predict(X_reshaped)
y_pred = np.argmax(y_pred_probs, axis=1)

# Print classification results
print("Classification Report:\n")
print(classification_report(y_encoded, y_pred, target_names=le.classes_))

print("Confusion Matrix:\n")
print(confusion_matrix(y_encoded, y_pred))
