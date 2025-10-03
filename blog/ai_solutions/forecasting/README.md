# Time Series Forecasting Demo (Streamlit + Holt-Winters)

This directory contains a minimal working time series forecasting demo using Streamlit with Holt-Winters Exponential Smoothing for trend and seasonality modeling.

## Quick start

* **Python 3.9+**
* **Install deps (CPU-friendly):**

```bash
pip install -U streamlit pandas numpy matplotlib statsmodels scikit-learn
streamlit run app.py
```

## Architecture

```
+-------------------+     +-------------------------+
| Streamlit UI      | <-> | Forecast Controller     |
| - Data upload     |     | (Session state)         |
| - Parameter tuning|     +------------+------------+
+-------------------+                  |
                                       v
                        +---------------------------+
                        | Holt-Winters ES Model     |
                        | - Trend component         |
                        | - Seasonal component      |
                        | - statsmodels backend     |
                        +-------------+-------------+
                                      |
                                      v
                        Forecast + Metrics + Plots
```

* **Data sources**: Synthetic (daily/monthly) or CSV upload
* **Model**: Triple Exponential Smoothing (Holt-Winters)
* **Evaluation**: MAE, RMSE, MAPE on test set
* **Visualization**: Training, test, forecast with confidence intervals

## Technical Summary

### Core Components

1. **Data Generation**
   - Synthetic daily sales with weekly seasonality (period=7)
   - Synthetic monthly traffic with yearly seasonality (period=12)
   - Custom CSV upload support

2. **Forecasting Model**
   - Exponential Smoothing with trend and seasonal components
   - Configurable trend type (additive, multiplicative, none)
   - Configurable seasonal type (additive, multiplicative, none)
   - Train/test split for validation

3. **Evaluation Metrics**
   - **MAE**: Mean Absolute Error
   - **RMSE**: Root Mean Squared Error
   - **MAPE**: Mean Absolute Percentage Error

4. **Visualization**
   - Historical data (training + test)
   - Test set predictions vs actuals
   - Future forecast with 95% confidence interval
   - Interactive matplotlib plots

## Files

* **app.py**: Streamlit app with data generation, model training, and forecasting
* **README.md**: This file with setup, architecture, and references

## Demo Behavior

* **Select data source**: Synthetic daily/monthly or upload CSV
* **Configure parameters**:
  - Forecast horizon (5-90 steps)
  - Test set size (10-40%)
  - Trend type (additive/multiplicative/none)
  - Seasonal type (additive/multiplicative/none)
* **Run forecast**: Click button to train model and generate predictions
* **View results**: Metrics, visualization, and forecast table
* **Confidence intervals**: 95% bounds based on test set variance

## References

* **Holt-Winters**: Winters, P. R. "Forecasting Sales by Exponentially Weighted Moving Averages" (Management Science, 1960)
* **statsmodels ExponentialSmoothing**: [https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html)
* **Time Series Forecasting**: Hyndman & Athanasopoulos. "Forecasting: Principles and Practice". [https://otexts.com/fpp3/](https://otexts.com/fpp3/)
* **Streamlit docs**: [https://docs.streamlit.io/](https://docs.streamlit.io/)

## Notes for Hugging Face Space Integration

* **Add requirements.txt** with pinned versions:
  ```
  streamlit==1.28.0
  pandas==2.0.3
  numpy==1.24.3
  matplotlib==3.7.2
  statsmodels==0.14.0
  scikit-learn==1.3.0
  ```
* **CPU-only deployment**: No GPU needed, lightweight model
* **Multi-demo integration**: Expose as tab in unified Space launcher
* **Production alternatives**:
  - Prophet (Facebook) for automatic seasonality detection
  - ARIMA/SARIMA for complex time series
  - LSTM/Transformer models for deep learning approaches
  - MLflow for experiment tracking and model registry
