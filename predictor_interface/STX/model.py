import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler


def load_model_and_scaler():
    model = tf.keras.models.load_model('ml_model/stock_model.h5')
    scaler = joblib.load('ml_model/scaler.pkl')
    return model, scaler


def train_model_and_save(df):
    df['Date'] = pd.to_datetime(df['Date'])

    # Feature Engineering
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_100'] = df['Close'].rolling(window=100).mean()
    df = df.dropna()

    features = df[['Close', 'MA_20', 'MA_100']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    seq_length = 30
    X_train, y_train = create_sequences(scaled_features, seq_length)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    model.save('ml_model/stock_model.h5')
    joblib.dump(scaler, 'ml_model/scaler.pkl')


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Predict the 'Close' price
    return np.array(X), np.array(y)


def forecast(model, input_data, n_steps, scaler, seq_length):
    forecasted = []
    current_input = input_data[-seq_length:].tolist()

    for _ in range(n_steps):
        input_array = np.array(current_input[-seq_length:]).reshape(1, seq_length, -1)
        next_value = model.predict(input_array)[0, 0]
        forecasted.append(next_value)
        current_input.append([next_value] + current_input[-1][1:])
        current_input.pop(0)

    forecasted = scaler.inverse_transform(
        np.concatenate((np.array(forecasted).reshape(-1, 1), np.zeros((n_steps, 2))), axis=1))[:, 0]
    return forecasted
