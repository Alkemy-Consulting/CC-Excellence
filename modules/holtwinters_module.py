import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def build_holtwinters_model(df, freq, periods_input=30, seasonal_periods=7, seasonal='add', trend='add'):
    model = ExponentialSmoothing(df['y'], trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    fit = model.fit()
    forecast_values = fit.forecast(periods_input)
    forecast_index = pd.date_range(start=df['ds'].max() + pd.tseries.frequencies.to_offset(freq), periods=periods_input, freq=freq)
    forecast_df = pd.DataFrame({'ds': forecast_index, 'yhat': forecast_values})
    return fit, forecast_df

def plot_forecast(df, forecast_df):
    df_plot = pd.concat([
        df[['ds', 'y']].set_index('ds'),
        forecast_df.set_index('ds')
    ]).reset_index()
    fig = px.line(df_plot, x='ds', y='y', title='Forecast Holt-Winters')
    fig.add_scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast')
    return fig

def evaluate_forecast(df, forecast_df):
    horizon = len(forecast_df)
    df_test = df[-horizon:]
    df_test = df_test.set_index('ds').reindex(forecast_df['ds']).dropna()
    y_true = df_test['y']
    y_pred = forecast_df.set_index('ds').loc[df_test.index]['yhat']

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return mae, rmse, mape, pd.DataFrame({'ds': y_true.index, 'y': y_true.values, 'yhat': y_pred.values})
