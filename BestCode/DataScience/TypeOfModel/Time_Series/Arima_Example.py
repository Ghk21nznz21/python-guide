""" ARIMA - Forecast --> Example """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from TimeSeries import test_stationarity, forecast_accuracy, correlations, ts_model, forward_chaining_CV
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv(
        'https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv', 
        names=['value'], header=0)
print(df)
plt.plot(df)

# Unit Root Test
test_stationarity(df.value.dropna())
test_stationarity(df.value.diff().dropna())

#Since P-value>0.05, take first differences and take a look to ACF plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Original Series
fig = correlations(df.value, max_lags=20)
fig.show()

# 1st Differencing
fig = correlations(df.value.diff(), max_lags=20)
fig.show()


# Arima - Residuals 
ts_model(
    ARIMA(df.value, order=(3,1,0)), df, 
    show_residuals_graph=True
)

ts_model(
    ARIMA(df.value, order=(3,1,0)), df, 
    show_residuals_graph=True
)


###### Forecasting 
train, test = forward_chaining_CV(df, 80)


# ARIMA 1 
arima1 = ARIMA(train, order=(3, 1, 0))  
tittle1 = "Forecast Arima(3,1,0)"
fig, arima1_fc = ts_model(arima1, train, test)
fig.title(tittle1)
fig.show()

print(tittle1, forecast_accuracy(arima1_fc, test.values))


# ARIMA 2
arima2 = ARIMA(train, order=(0, 1, 3))  
tittle2 = "Forecast Arima(0,1,3)"
fig2, arima2_fc = ts_model(arima2, train, test)
fig2.title(tittle2)
fig2.show()
print(tittle2, forecast_accuracy(arima2_fc, test.values))


# Forecast the future
model = ARIMA(df, order=(3, 1, 0))  
fig3, fc = ts_model(model, df, 20)
plt.title('20-Step Ahead Forecasts for Internet Users using the ARIMA(0,1,3)')
plt.legend(loc='upper left', fontsize=8)
plt.show()

