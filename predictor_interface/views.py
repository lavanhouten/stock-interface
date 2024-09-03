from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import io
from io import BytesIO
import base64
from .STX.model import create_sequences, forecast, load_model_and_scaler, train_model_and_save
from datetime import datetime


def index(request):
    return render(request, 'interface/index.html')

def train_model(request):
    if request.method == "POST":
        file = request.FILES['train_file']
        df = pd.read_csv(file)
        train_model_and_save(df)
        return render(request, 'interface/train_result.html', {'message': 'Model trained successfully!'})
    return render(request, 'interface/train.html')

def predict_model(request):
    if request.method == "POST":
        file = request.FILES['predict_file']
        df = pd.read_csv(file)
        plot_url = predict_stock_price(df)
        return render(request, 'interface/result.html', {'plot_url': plot_url})
    return render(request, 'interface/predict.html')

def predict_stock_price(df):
    # Add your model loading and prediction code here
    model, scaler = load_model_and_scaler()
    df['Date'] = pd.to_datetime(df['Date'])

    # Feature Engineering
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    df = df.dropna()

    features = df[['Close', 'MA_50', 'MA_200']].values
    scaled_features = scaler.transform(features)

    seq_length = 60
    test_dates = df['Date'].values[seq_length:]

    X_test, y_test = create_sequences(scaled_features, seq_length)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 2))), axis=1))[:, 0]
    y_test_scaled = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 2))), axis=1))[:, 0]

    # Plot results with Plotly
    actual_trace = go.Scatter(
        x=test_dates,
        y=y_test_scaled,
        mode='lines',
        name='Actual Stock Price'
    )

    predicted_trace = go.Scatter(
        x=test_dates,
        y=predictions,
        mode='lines',
        name='Predicted Stock Price'
    )

    n_steps = 30
    last_input = scaled_features[-seq_length:]
    last_date = test_dates[-1]
    future_dates = pd.date_range(start=last_date, periods=n_steps + 1, freq='D')[1:]
    future_predictions = forecast(model, last_input, n_steps, scaler, seq_length)

    future_trace = go.Scatter(
        x=future_dates,
        y=future_predictions,
        mode='lines',
        name='Future Predictions'
    )

    layout = go.Layout(
        title='Stock Price Prediction and Future Projections',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Stock Price')
    )

    fig = go.Figure(data=[actual_trace, predicted_trace, future_trace], layout=layout)
    plot_div = pio.to_html(fig, full_html=False)
    
    return plot_div