import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_and_forecast_prophet(df, freq='D', periods=30, use_holidays=False, yearly=True, weekly=False, daily=False, seasonality_mode='additive', changepoint_prior_scale=0.05):
    """
    Build and forecast using a Prophet model.

    Parameters:
        df (pd.DataFrame): DataFrame with 'ds' (datetime) and 'y' (target) columns.
        freq (str): Frequency of the forecast (e.g., 'D' for daily).
        periods (int): Number of periods to forecast.
        use_holidays (bool): Whether to include holiday effects.
        yearly (bool): Enable yearly seasonality.
        weekly (bool): Enable weekly seasonality.
        daily (bool): Enable daily seasonality.
        seasonality_mode (str): 'additive' or 'multiplicative'.
        changepoint_prior_scale (float): Flexibility of the trend.

    Returns:
        model (Prophet): Trained Prophet model.
        forecast (pd.DataFrame): Forecasted values.
    """
    holidays = None
    if use_holidays:
        years = df['ds'].dt.year.unique()
        holiday_dates = [f"{year}-12-25" for year in years]  # Example: Christmas
        holidays = pd.DataFrame({'ds': pd.to_datetime(holiday_dates), 'holiday': 'holiday'})

    model = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=daily,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        holidays=holidays
    )

    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return model, forecast


def evaluate_forecast(df, forecast):
    """
    Evaluate forecast accuracy.

    Parameters:
        df (pd.DataFrame): Actual data with 'ds' and 'y'.
        forecast (pd.DataFrame): Forecasted data with 'ds' and 'yhat'.

    Returns:
        dict: Dictionary with MAE, RMSE, MAPE, and combined DataFrame.
    """
    df_forecast = forecast[['ds', 'yhat']].set_index('ds')
    df_actual = df.set_index('ds')
    df_combined = df_actual.join(df_forecast, how='left').dropna()

    mae = mean_absolute_error(df_combined['y'], df_combined['yhat'])
    mse = mean_squared_error(df_combined['y'], df_combined['yhat'])
    rmse = mse ** 0.5
    mape = np.mean(np.abs((df_combined['y'] - df_combined['yhat']) / df_combined['y'])) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'combined': df_combined
    }


def plot_forecast(model, forecast):
    """
    Plot the forecast using Plotly.

    Parameters:
        model (Prophet): Trained Prophet model.
        forecast (pd.DataFrame): Forecasted values.

    Returns:
        plotly.graph_objs.Figure: Plotly figure.
    """
    return plot_plotly(model, forecast)


def plot_components(model, forecast):
    """
    Plot the forecast components using Plotly.

    Parameters:
        model (Prophet): Trained Prophet model.
        forecast (pd.DataFrame): Forecasted values.

    Returns:
        plotly.graph_objs.Figure: Plotly figure.
    """
    return plot_components_plotly(model, forecast)


def run_prophet_model(df, date_col, target_col, freq, horizon, make_forecast):
    """
    Run the Prophet model and display results in the main page.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        date_col (str): Name of the date column.
        target_col (str): Name of the target column.
        freq (str): Frequency of the forecast.
        horizon (int): Number of periods to forecast.
        make_forecast (bool): Whether to make a forecast.

    Returns:
        None
    """
    st.subheader("Prophet Forecast")

    # Prepare data for Prophet
    prophet_df = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})

    # Build and forecast
    model, forecast = build_and_forecast_prophet(
        prophet_df,
        freq=freq,
        periods=horizon if make_forecast else 0
    )

    # Use a container to display results
    with st.container():
        # Display forecast plot
        st.plotly_chart(plot_forecast(model, forecast), use_container_width=True)

        # Display components plot
        st.plotly_chart(plot_components(model, forecast), use_container_width=True)

        # Evaluate and display metrics
        metrics = evaluate_forecast(prophet_df, forecast)
        st.write("### Evaluation Metrics")
        st.write({key: metrics[key] for key in ['MAE', 'RMSE', 'MAPE']})
