import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Time Series Forecasting Demo", layout="wide")

st.title("📈 Time Series Forecasting Demo")
st.markdown(
    """A minimal working time series forecasting demo using Exponential Smoothing (Holt-Winters) 
    for trend and seasonality modeling. Upload your own data or use synthetic samples."""
)

# Sidebar
st.sidebar.header("Forecast Settings")
data_source = st.sidebar.selectbox("Data Source", ["Synthetic - Daily Sales", "Synthetic - Monthly Traffic", "Upload CSV"])

# Generate or load data
@st.cache_data
def generate_daily_sales(n_days=365):
    """Generate synthetic daily sales with trend and weekly seasonality"""
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    trend = np.linspace(100, 150, n_days)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # Weekly pattern
    noise = np.random.normal(0, 5, n_days)
    sales = trend + seasonality + noise
    return pd.DataFrame({'date': dates, 'value': sales}).set_index('date')

@st.cache_data
def generate_monthly_traffic(n_months=48):
    """Generate synthetic monthly traffic with trend and yearly seasonality"""
    dates = pd.date_range(start='2020-01-01', periods=n_months, freq='MS')
    trend = np.linspace(1000, 2000, n_months)
    seasonality = 300 * np.sin(2 * np.pi * np.arange(n_months) / 12)  # Yearly pattern
    noise = np.random.normal(0, 50, n_months)
    traffic = trend + seasonality + noise
    return pd.DataFrame({'date': dates, 'value': traffic}).set_index('date')

if data_source == "Synthetic - Daily Sales":
    df = generate_daily_sales()
    freq = 'D'
    seasonal_periods = 7
    st.sidebar.info("Daily sales data with weekly seasonality")
elif data_source == "Synthetic - Monthly Traffic":
    df = generate_monthly_traffic()
    freq = 'MS'
    seasonal_periods = 12
    st.sidebar.info("Monthly traffic data with yearly seasonality")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV (columns: date, value)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=['date'], index_col='date')
        freq = st.sidebar.selectbox("Frequency", ['D', 'W', 'MS', 'Q', 'Y'])
        seasonal_periods = st.sidebar.number_input("Seasonal Periods", min_value=2, value=7)
    else:
        st.warning("Please upload a CSV file with 'date' and 'value' columns")
        st.stop()

# Forecasting parameters
st.sidebar.subheader("Model Parameters")
forecast_steps = st.sidebar.slider("Forecast Horizon", min_value=5, max_value=90, value=30)
test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=40, value=20)
trend_type = st.sidebar.selectbox("Trend", ['add', 'mul', None])
seasonal_type = st.sidebar.selectbox("Seasonality", ['add', 'mul', None])

# Split data
split_point = int(len(df) * (1 - test_size / 100))
train = df.iloc[:split_point]
test = df.iloc[split_point:]

st.sidebar.metric("Training samples", len(train))
st.sidebar.metric("Test samples", len(test))

# Data overview
st.subheader("Data Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Observations", len(df))
with col2:
    st.metric("Mean Value", f"{df['value'].mean():.2f}")
with col3:
    st.metric("Std Dev", f"{df['value'].std():.2f}")

# Show data sample
if st.checkbox("Show raw data"):
    st.dataframe(df.head(20), use_container_width=True)

# Train model
st.subheader("Model Training & Forecasting")

if st.button("Run Forecast", type="primary"):
    with st.spinner("Training model..."):
        try:
            # Fit Holt-Winters model
            if seasonal_type:
                model = ExponentialSmoothing(
                    train['value'],
                    trend=trend_type,
                    seasonal=seasonal_type,
                    seasonal_periods=seasonal_periods
                )
            else:
                model = ExponentialSmoothing(
                    train['value'],
                    trend=trend_type
                )
            
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=len(test) + forecast_steps)
            forecast_test = forecast[:len(test)]
            forecast_future = forecast[len(test):]
            
            # Calculate metrics on test set
            mae = mean_absolute_error(test['value'], forecast_test)
            rmse = np.sqrt(mean_squared_error(test['value'], forecast_test))
            mape = np.mean(np.abs((test['value'] - forecast_test) / test['value'])) * 100
            
            # Display metrics
            st.success("Model trained successfully!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{mae:.2f}")
            with col2:
                st.metric("RMSE", f"{rmse:.2f}")
            with col3:
                st.metric("MAPE", f"{mape:.2f}%")
            
            # Visualization
            st.subheader("Forecast Visualization")
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot training data
            ax.plot(train.index, train['value'], label='Training Data', color='blue', linewidth=2)
            
            # Plot test data
            ax.plot(test.index, test['value'], label='Test Data (Actual)', color='green', linewidth=2)
            
            # Plot test forecast
            ax.plot(test.index, forecast_test, label='Test Forecast', color='orange', 
                   linewidth=2, linestyle='--')
            
            # Plot future forecast
            future_dates = pd.date_range(start=df.index[-1], periods=forecast_steps+1, freq=freq)[1:]
            ax.plot(future_dates, forecast_future, label='Future Forecast', 
                   color='red', linewidth=2, linestyle='--')
            
            # Add confidence interval (simplified)
            if len(forecast_future) > 0:
                std_dev = test['value'].std()
                ax.fill_between(future_dates, 
                              forecast_future - 1.96*std_dev, 
                              forecast_future + 1.96*std_dev,
                              alpha=0.2, color='red', label='95% Confidence Interval')
            
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=12, fontweight='bold')
            ax.set_title('Time Series Forecast', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Forecast table
            st.subheader("Future Forecast Values")
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecasted Value': forecast_future.values
            })
            st.dataframe(forecast_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Forecasting error: {str(e)}")
            st.info("Try adjusting model parameters or check your data quality")

# Footer
st.markdown("---")
st.caption(
    "This demo uses Holt-Winters Exponential Smoothing from statsmodels. "
    "For production systems, consider Prophet, ARIMA, LSTM, or Transformer-based models."
)
