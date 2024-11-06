### Name:Karnan K
### Reg No:212222230062
### Date: 
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL


### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset Airline Baggage
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima


df = pd.read_csv('airline_baggage_complaints.csv', parse_dates=['Date'], index_col='Date')


date_range = pd.date_range(start='2020-01-01', periods=100, freq='M')
complaints = np.random.poisson(lam=20, size=100)
df = pd.DataFrame({'Date': date_range, 'Complaints': complaints}).set_index('Date')

# Visualize the data
plt.figure(figsize=(10, 6))
plt.plot(df, label="Baggage Complaints")
plt.title("Airline Baggage Complaints Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Complaints")
plt.legend()
plt.show()


result = adfuller(df['Complaints'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])


df_diff = df.diff().dropna() if result[1] > 0.05 else df

# Plot ACF and PACF for SARIMA parameters
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_acf(df_diff, ax=plt.gca(), lags=20)
plt.title("ACF Plot")
plt.subplot(1, 2, 2)
plot_pacf(df_diff, ax=plt.gca(), lags=20)
plt.title("PACF Plot")
plt.tight_layout()
plt.show()


seasonal_period = 12
auto_model = auto_arima(df, seasonal=True, m=seasonal_period, trace=True)
p, d, q = auto_model.order
P, D, Q, s = auto_model.seasonal_order
print(f"SARIMA order: ({p}, {d}, {q}) x ({P}, {D}, {Q}, {s})")


sarima_model = SARIMAX(df, order=(p, d, q), seasonal_order=(P, D, Q, s))
sarima_fit = sarima_model.fit(disp=False)
print(sarima_fit.summary())


forecast = sarima_fit.get_forecast(steps=12)
forecast_index = pd.date_range(df.index[-1] + pd.DateOffset(1), periods=12, freq='M')
forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)

# Plot actual data and forecasted values
plt.figure(figsize=(10, 6))
plt.plot(df, label="Actual Complaints")
plt.plot(forecast_series, label="Forecasted Complaints", color="orange")
plt.title("Forecasted Airline Baggage Complaints")
plt.xlabel("Date")
plt.ylabel("Number of Complaints")
plt.legend()
plt.show()

# Evaluate Model Performance
# Use the last 20 data points as test data for evaluation
y_train = df.iloc[:-20]
y_test = df.iloc[-20:]
sarima_fit_train = SARIMAX(y_train, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=False)
y_pred = sarima_fit_train.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot actual vs predicted on test set
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Actual Complaints")
plt.plot(y_pred, label="Predicted Complaints", color="red")
plt.title("Model Evaluation on Test Data")
plt.xlabel("Date")
plt.ylabel("Number of Complaints")
plt.legend()
plt.show()
```
### OUTPUT:

![download](https://github.com/user-attachments/assets/8b484fd7-f55a-47f9-9ac5-96cdfdf06a6d)


## ACF AND PACF

![download](https://github.com/user-attachments/assets/7400f918-2f64-4c8f-8360-9eb2626cd2ee)

## actual and forecasted values

![download](https://github.com/user-attachments/assets/f1749b6d-2b19-4dfd-9fb9-0631ae94bb9b)

## actual vs predicted

![download](https://github.com/user-attachments/assets/dbc679b0-878d-489e-bd27-d0de5a66e291)

### RESULT:
Thus the program run successfully based on the SARIMA model.
