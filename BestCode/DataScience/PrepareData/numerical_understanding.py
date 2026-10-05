import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor, ceil


class NumericalAnalyses:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.originial_df = self.df.copy()

    def steps(self):
        '''
        pd.Describe
        sns.Pairplots
        pd.Boxplots
        '''

    def show_boxplots(self, col: str):
        '''
        Show boxplot with statistics
        manually writen.
        '''
        plt.figure(figsize=(16, 10))
        sns.boxplot(self.df[col], orient='h')
        plt.axvline(x=self.df[col].mean(), color='g', linestyle='--')

        text_metada = [
            {'statistics': 'mean', 'value': 0.48, 'pos': 'center'},
            {'statistics': 'median', 'value': -0.42, 'pos': 'center'},
            {'statistics': 'max', 'value': 0.48, 'pos': 'right'},
            {'statistics': 'min', 'value': 0.48, 'pos': 'left'}
        ]
        # IF col == ' Annual_Premium' -> increase value by 500
        for text_specifics in text_metada:
            function_reference = getattr(
                pd.Series, text_specifics['statistics'], None)
            message = f"{text_specifics['statistics'].capitalize()}: \
                    {self.df[col].mean():.2f}"
            plt.text(
                function_reference(self.df[col]),
                text_specifics['value'], message,
                ha='center', fontsize=16)

        plt.title(f'{col} Distribution in the Dataset')
        plt.xlabel(col)
        plt.show()

    def hist(self, column: str, bins=10, **kwargs):
        sns.histplot(self.df[column], bins=10, **kwargs)

    def pairplots(self, col1: str, col2: str):
        # sns.pairplot / df.hvplot.hist(y=col1, by=col2, bins=30)
        pass

    def corr(self, numvars: list):
        sns.heatmap(self.df[numvars].corr(), annot=True, fmt=".2f")
        plt.show()


target = 'Response'
cat_columns = [
    'id', 'Gender', 'Driving_License',
    'Region_Code', 'Previously_Insured', 'Vehicle_Age',
    'Vehicle_Damage', 'Policy_Sales_Channel', target
]
'''
numericals = ['Annual_Premium', 'Age', 'Vintage']
df_original = pd.read_csv('train.csv')
numerical_class = NumericalAnalyses(
    df_original.loc[:, numericals])

numerical_class.hist('Annual_Premium')
'''
