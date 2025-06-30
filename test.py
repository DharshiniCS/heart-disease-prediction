import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, SimpleRNN, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

df = pd.read_csv("csv_dataset/training-dataset_echo.csv")
X = df.drop(['label', 'stage', 'treatment', 'mortality'], axis=1).select_dtypes(include=[np.number]).values
y = df['label'].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_cat, test_size=0.2, random_state=42)
model = Sequential([
    Conv1D(64, kernel_size=2, activation='relu', padding='same'),
    MaxPooling1D(1),
    SimpleRNN(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1)
model.save("cnn_rnn_model.keras")
np.save("label_encoder_classes.npy", le.classes_)
joblib.dump(scaler, "scaler.save")
