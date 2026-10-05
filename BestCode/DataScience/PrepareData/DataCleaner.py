import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import expon, kstest

class DataCleaner:
    """ Handle all cleaning operations """

    def __init__(self, data: pd.DataFrame, target: str):
        """ 
        Initialize the DataCleaner object.

        @param data: DataFrame to be cleaned
        @param target: Name of the target column
        """
        self.data = data
        self.X_cols = [col for col in self.data.columns if col != target]
        self.target = target

    def handle_missing_values_except_target(self):
        """ Handle missing values by removing rows with missing values, excluding the target column. """
        self.data.dropna(subset=self.X_cols, inplace=True)

    def handle_outliers(self, numerical_vars: list, distance=1.5, remove_outliers=False, use_log=False):
        """
        Handle outliers using interquartile range (IQR) on specified numerical variables.

        @param numerical_vars: List of numerical variables to handle outliers
        @param distance: Multiplier for IQR to define outliers
        @param remove_outliers: Whether to remove outliers from the data (default: False)
        @param use_log: Whether to base analysis on raw or log-transformed data (default: False).
        """
        for var in numerical_vars:
            if use_log:
                non_zero_values = self.data[var] > 0
                log_var = np.log(self.data[var][non_zero_values])
                q1 = log_var.quantile(0.25)
                q3 = log_var.quantile(0.75)
            else:
                q1 = self.data[var].quantile(0.25)
                q3 = self.data[var].quantile(0.75)

            iqr = q3 - q1
            lower_threshold = q1 - distance * iqr
            upper_threshold = q3 + distance * iqr

            if remove_outliers:
                if use_log:
                    self.data = self.data[(non_zero_values & (log_var >= lower_threshold) & (log_var <= upper_threshold)) | (self.data[var] == 0)]
                else:
                    self.data = self.data[(self.data[var] >= lower_threshold) & (self.data[var] <= upper_threshold)]
            else:
                if use_log:
                    self.data.loc[(non_zero_values & (log_var < lower_threshold) & (self.data[var] != 0)), var] = np.exp(lower_threshold)
                    self.data.loc[(non_zero_values & (log_var > upper_threshold) & (self.data[var] != 0)), var] = np.exp(upper_threshold)
                else:
                    self.data.loc[self.data[var] < lower_threshold, var] = lower_threshold
                    self.data.loc[self.data[var] > upper_threshold, var] = upper_threshold

    def convert_to_dummies(self, categorical_vars: list):
        """ Convert categorical variables into dummy variables. """
        self.data = pd.get_dummies(self.data, columns=categorical_vars, drop_first=True, dtype=int)
        
    def encoding(self, categorical_cols: list):
        ''' Save Encoding maps  '''
        self.encoders = {}
        for col in categorical_cols: 
            enc = LabelEncoder()
            self.df[col] = enc.fit_transform(self.df[col]).astype('str')
            encoding_map = {
                str(encoded_value): class_name for class_name, encoded_value 
                in zip(enc.classes_, enc.transform(enc.classes_))}
            self.encoders[col] = encoding_map
            
            

    def standardize_vars(self, variables: list):
        """ Standardize numerical or ordinal variables using MinMaxScaler. """
        scaler = MinMaxScaler()
        self.data[variables] = scaler.fit_transform(self.data[variables])

    def test_exponential_distribution(self, variable: str):
        """ Test if the data follows an exponential distribution using the Kolmogorov-Smirnov test. """
        data = self.data[variable].copy()
        if variable == self.target:
            data = data.dropna()
        scale = expon.fit(data)[1]
        test_stat, p_value = kstest(data, 'expon', args=(0, scale))
        print("Exponential Goodness-of-Fit Test:")
        print("Test Statistic:", test_stat)
        print("p-value:", p_value)

    def transform_numerical(self, variables: list, transformation="None"):
        """
        Perform transformations on numerical variables.

        @param variables: List of variables to transform
        @param transformation: Transformation type ('None', 'Log', 'SQRT', 'MIX')
        """
        valid_transforms = {"None", "Log", "SQRT", "MIX"}
        
        if transformation not in valid_transforms:
            raise ValueError("Invalid transformation type. Supported types: " + ", ".join(valid_transforms))
        
        for var in variables:
            if transformation == "Log":
                non_zero_values = self.data[var] > 0
                self.data[var][non_zero_values] = np.log(self.data[var][non_zero_values])
            elif transformation == "SQRT":
                self.data[var] = np.sqrt(self.data[var])
            elif transformation == "MIX":
                self.data[var] = self.data[var].apply(lambda x: x if x == 0 else np.log(x))
