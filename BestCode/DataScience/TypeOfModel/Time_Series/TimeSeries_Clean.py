import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class Preprocess:
    def __init__(self, data):
        self.data = data
        self.correct_names()
        self.data_size = self.data.columns
        self.target_name = self.data.columns[-1]
        self.correct_names()
        self.transformation_measures = {}

    def correlations(self, cols, max_lags):
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(cols, lags=max_lags, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(cols, lags=max_lags, ax=ax2)
        return fig

    def test_stationarity(self, timeseries, lags=None):
        # Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test:')
        if lags:
            dftest = adfuller(timeseries, lags=lags)
        else:
            dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(
            dftest[0:4],
            index=['Test Statistic', 'p-value',
                   '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)

    def correct_names(self):
        self.data.columns = [name.replace(' ', '_') for name in self.data.columns]

    def reupdate_cols(self):
        ''' every time main dataframe columns are change'''
        self.data_size = self.data.columns
        self.target_name = self.data.columns[-1]
        self.correct_names()

    def psi_index(self, normalize=False):
        corr = np.corrcoef(self.data, rowvar=False)

        # Eigenvalues and eigenvectors from the correlation matrix
        eig_val, eig_vec = np.linalg.eig(corr)
        idx = eig_val.argsort()[::-1]
        eig_val = eig_val[idx]
        if normalize:
            p = self.data.shape[1]
            return round(np.sum((eig_val - 1)**2) / (p*(p-1)), 5)
        else:
            return round(np.sum((eig_val - 1)**2), 5)

    def scatter_coefficient(self, normalize=True):
        corr = np.corrcoef(self.data, rowvar=False)
        if normalize:
            return round(np.linalg.det(corr) / np.trace(corr), 5)
        else:
            return round(np.linalg.det(corr), 5)

    def visualize(self, label):
        nrows = len(self.data_size)/2
        nrows = math.ceil(nrows)
        fig, ax = plt.subplots(nrows=nrows, ncols=2, dpi=120, figsize=(5*nrows, 20))
        index = 0
        for row in range(nrows):
            for col in range(2):
                ax[row, col].plot(self.data.iloc[:, index], color='red', linewidth=1)
                ax[row, col].set_title(str(self.data.columns[index]), fontsize=16)
                ax[row, col].axhline(0, linestyle='--', color='k', alpha=0.3)
                ax[row, col].set_ylabel(label, fontsize=16)
                index += 1

    def normalize(self):
        for col in self.data_size:
            avg, std = self.data[col].mean(), self.data[col].std()
            self.data[col] = (self.data[col] - avg)/std
            if col == self.target_name:
                self.transformation_measures['normalize'] = [avg, std]

    def stationary(self):
        '''
        If its stationary then its mean, variance and auto-covariance are independent of time.
        Which implies that has no trend or seasonality
        '''
        for col in self.data_size:
            dftest = adfuller(self.data[col], autolag='AIC')
            output = f'Indicator: {col}\nADF:{dftest[0]}\nP-Value: {dftest[1]}\nNº Lags: {dftest[2]}'
            print(output)
            for key, val in dftest[4].items():
                print("\t", key, ": ", val)
            print('\n \n')
        # p-value > 0.05: Fail to reject -> non-stationary.
        # p-value <= 0.05: Reject  -> stationary.

    def remove_trend(self):
        for col in self.data_size:
            if col == self.target_name:
                self.transformation_measures['trend'] = self.data[col]
            self.data[col] = self.data[col].diff().dropna()
        self.data.dropna(axis=0, how='all', inplace=True)

    def remove_volatility_increase(self):
        for col in self.data_size:
            year_vol = self.data[col].groupby(self.data[col].index.year).std()
            year_vol = self.data[col].index.map(lambda d: year_vol.loc[d.year])
            if col == self.target_name:
                self.transformation_measures['volatility'] = year_vol.values
            self.data[col] = self.data[col]/year_vol
        self.data.dropna(axis=0, how='all', inplace=True)

    def remove_seasonality(self):
        for col in self.data_size:
            month = self.data[col].groupby(self.data[col].index.month).mean()
            month = self.data[col].index.map(lambda d: month.loc[d.month])
            if col == self.target_name:
                self.transformation_measures['seasonality'] = month.values
            self.data[col] = self.data[col] - month
        self.data.dropna(axis=0, how='all', inplace=True)

    def autocorrelation_plots(self):
        plot_pacf(self.data.iloc[:, -1])
        plt.show()

        plot_acf(self.data.iloc[:, -1])
        plt.show()


def invert_transformations(target, transformation_measures):
    size = target.size
    # add seasonality
    target.loc[:, 'Pct'] += transformation_measures['seasonality'][-size:]
    # add volatility
    target.loc[:, 'Pct'] = target.loc[:, 'Pct'] * transformation_measures['volatility'][-size:]
    # add first diff
    pre_diff = transformation_measures['trend'].shift(1)
    target.loc[:, 'Pct'] = pre_diff[-size:] + target.loc[:, 'Pct']
    # denormalize
    target = target * transformation_measures['normalize'][1] + transformation_measures['normalize'][0]
    return target
