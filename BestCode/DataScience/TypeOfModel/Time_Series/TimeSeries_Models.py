'''
Plot Data

Stationary:
    Correlations: ACF + PACF 
    Test: Dickey-Fuller
    
    Remove: APPLY DIFF
        a) Apply Differences Log and Normal 
                (trend diff, seasonal diff, trend and seasonal diff)
            data[col] = data[col].shift(lag_number)
        b) Decomposition 
            from statsmodels.tsa.seasonal import seasonal_decompose
        c) Holt's method
            from statsmodels.tsa.holtwinters \
                import ExponentialSmoothing, SimpleExpSmoothing
    Check again: ACF + PACF 

Residuals?
Model
Forecast - Evaluation Metrics

''' 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import (ExponentialSmoothing,
                                         SimpleExpSmoothing)
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, pacf

sm.tsa.statespace.SARIMAX

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential


def correlations(cols: list[list, ] | list, max_lags):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(cols, lags=max_lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(cols, lags=max_lags, ax=ax2)
    return fig 

def test_stationarity(timeseries: list, lags=None):
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    if lags:
        dftest = adfuller(timeseries, lags=lags)
    else:
        dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index=['Test Statistic','p-value',
                                '#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    

def forward_chaining_CV(data: pd.DataFrame, test_samples:int):
    ''' Based on forecasting division 
        :returns train, test 
    '''
    return  data.loc[:test_samples], data.loc[test_samples:]

def forecast_accuracy(forecast, actual):
    ''' Can also use sklearn.metrics 
    '''
    mape = round(np.mean(np.abs(forecast - actual)/np.abs(actual))*100 , 2)  # MAPE
    mae = round(np.mean(np.abs(forecast - actual)) , 2) # MAE
    rmse = round(np.mean((forecast - actual)**2)**.5 , 2) # RMSE
    return({'mape(%)':mape, 'mae': mae, 'rmse':rmse})

    return 

def arima(model: object, train: pd.DataFrame, test: pd.DataFrame | int | None,
            show_residuals_graph = False):
    '''
    ARIMA:
    	CORR
    	STATIONARY  
    test: can be either a dataframe to check metrics
        or int for actual future predictions 
    
    '''
    def plot_residuals():
        # Plot residual errors
        residuals = pd.DataFrame(model.resid)
        fig, axes = plt.subplots()  
        axes.plot(residuals)
        axes.set_title("Residuals")
        fig, axes = plt.subplots(1, 2, sharex=True)
        correlations(residuals.dropna(), max_lags=20)
        plt.show()
        return 
    
    print(model.summary())  
    model = model.fit()
    if show_residuals_graph:
        plot_residuals()
        
    if test is None: 
        return 
    # apply forecast     
    fc = model.forecast(test) if type(test) == int else model.forecast(test.shape[0])
    forecast = pd.Series(fc, index=test.index)
    
    # Plot Forecast 
    fig = plt.figure(figsize=(12,5), dpi=100)
    plt.plot(train, label='training')
    if type(test) == pd.DataFrame:
        plt.plot(test, label='actual')
    plt.plot(forecast, label='forecast')
    #plt.title('...')
    plt.legend(loc='upper left', fontsize=8)
    return fig, forecast




class Models:
    pass 
