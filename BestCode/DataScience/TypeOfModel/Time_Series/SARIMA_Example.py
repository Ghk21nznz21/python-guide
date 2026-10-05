""" SARIMA """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm  

from TimeSeries import test_stationarity, forecast_accuracy, correlations


url = "https://raw.githubusercontent.com/PinkWink/DataScience/master/data/07.%20portland-oregon-average-monthly-.csv" 
df = pd.read_csv(url, index_col='Month',parse_dates=True)
df.index.freq = 'MS'
df.columns= ['riders']
print(df)

# Plot data
df.riders.plot(figsize=(12,8), title= 'Monthly Ridership', fontsize=14)

correlations(df.riders, max_lags=36)

test_stationarity(df.riders)  

#######     DIFFERENCES

# Log of Riders
df.riders_log= df.riders.apply(lambda x: np.log(x))  
test_stationarity(df.riders_log)
df.riders_log.plot(figsize=(12,8), title= 'Logarithm of riders', fontsize=14)

# First differences
df['first_difference'] = df.riders - df.riders.shift(1)  
test_stationarity(df.first_difference.dropna(inplace=False))
df['first_difference'].plot(figsize=(12,8), 
                            title= 'First difference', fontsize=14)

# Differences of Log
df['log_first_difference'] = df.riders_log - df.riders_log.shift(1)  
test_stationarity(df.log_first_difference.dropna(inplace=False))


# Sesaonal Differences
df['seasonal_difference'] = df.riders - df.riders.shift(12)  # month -> year 
test_stationarity(df.seasonal_difference.dropna(inplace=False))
df['seasonal_difference'].plot(figsize=(12,8), 
                               title= 'Sesaonal difference', fontsize=14)


# Ordinary and seasonal differences
df['seasonal_first_difference'] = df.first_difference - df.first_difference.shift(12)  
test_stationarity(df.seasonal_first_difference.dropna(inplace=False))
df['seasonal_first_difference'].plot(figsize=(12,8), 
                                     title= 'Ordinary and seasonal differences', fontsize=14)


# Ordinary and seasonal differences of logs
df['log_seasonal_first_difference'] = df.log_first_difference - df.log_first_difference.shift(12)  
test_stationarity(df.log_seasonal_first_difference.dropna(inplace=False))
df['log_seasonal_first_difference'].plot(figsize=(12,8), 
                                     title= 'Ordinary and seasonal differences of logarithm of riders', fontsize=14)



#####################
# ACF and PACF of stationary time series: Ordinary and seasonal differences
# Do not include the first 13 observations
correlations(df.seasonal_first_difference.iloc[13:], max_lags=36)

###########################
# Fit a SARIMA(0,1,0)(0,1,1)12 model
model1 = sm.tsa.statespace.SARIMAX(df.riders, trend='n', 
                                order=(0,1,0), 
                                seasonal_order=(0,1,1,12))
results1 = model1.fit()
print(results1.summary())

#########################
# Fit a SARIMA(0,1,0)(1,1,0)12 model
model2 = sm.tsa.statespace.SARIMAX(df.riders, trend='n', 
                                order=(0,1,0), 
                                seasonal_order=(1,1,0,12))
results2 = model2.fit()
print(results2.summary())

################# Forecast 
df['forecast'] = results1.predict(start = 102, end = 114, dynamic= True)  
df[['riders', 'forecast']].plot(figsize=(12, 8)) 
test = df.iloc[-12:] #last 12 observations
print(forecast_accuracy(test['riders'], test['forecast']))








  