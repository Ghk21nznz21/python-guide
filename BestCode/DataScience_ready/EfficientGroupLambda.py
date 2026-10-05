import pandas as pd 
import numpy as np 

class EfficentGroupLambda:
    ''' 
    Most Usefull:
        map: element wise on series
        apply:  element wise on series 
        applymap: element wise on Dataframe
        tranform: element-wise transformations (often with groupby)
        pipe: sequence of functions 
        aggregate: aggregation function (like sum, mean, ...)
    '''
    @staticmethod
    def fastRead(path):
        return pd.concat(
            pd.read_csv(path, iterator=True, chunksize=1000), 
            ignore_index=True
        )

    @staticmethod
    def lambda_on_groups(df: pd.DataFrame, group_cols: list[str, ], myLambda: object):
        ''' Can also be just to change one collum'''
        return df.groupby(group_cols).transform(myLambda)

    @staticmethod
    def multiple_columns_shift(df, group_cols, apply_cols, shifts): # Faster 
        ''' Applies multiple column shifts within groups
        @param df: The input DataFrame.
        @param group_cols : Columns to use for grouping the DataFrame.
        @param apply_cols : Columns to which the shifting transformation will be applied.
        @param n_shifts: Number of shifts to be applied to each column.

        @return: DataFrame with new columns representing shifted values.
        '''
        def col_shifts(series, shifts):
            return pd.concat({f"{series.name}_shift{i}": series.shift(-i) for i in shifts}, axis=1)

        new_cols = df.groupby(group_cols)[apply_cols].apply(
            lambda group: col_shifts(group.iloc[:, 0], shifts)
        )
        return pd.concat([df, new_cols], axis=1)
    
    @staticmethod
    def multiple_columns_shift(df: pd.DataFrame, group_cols: list, apply_cols: list, shifts: list) -> pd.DataFrame:
        ''' Applies multiple column shifts within groups.

        @param df: The input DataFrame.
        @param group_cols : Columns to use for grouping the DataFrame.
        @param apply_cols : Columns to which the shifting transformation will be applied.
        @param n_shifts: Number of shifts to be applied to each column.

        @return: DataFrame with new columns representing shifted values.
        
        Notes:
            Due to Groupby renaming columns is faster then .rename 
            Due to converting multiple columns with the same function, looping trought shifts is faster
                then using apply or transform is faster 
        ''' 
        shift_dataframes = []
        for i in shifts: 
            shifted_cols = df.groupby(group_cols)[apply_cols].shift(-i)
            new_col_names = [f"{col}_shift{i}" for col in apply_cols]
            shifted_cols.columns = new_col_names
            shift_dataframes.append(shifted_cols)
        return pd.concat([df, *shift_dataframes], axis=1)




# Create a sample DataFrame
np.random.seed(42)
data = {
    'GroupCol1': np.random.choice(['A', 'B', 'C'], size=100),
    'GroupCol2': np.random.choice(['X', 'Y'], size=100),
    'ApplyCol1': np.random.rand(100),
    'ApplyCol2': np.random.rand(100)
}

df = pd.DataFrame(data)

# Define the shifts and group_cols
shifts = [1, 2, 3]
group_cols = ['GroupCol1', 'GroupCol2']
apply_cols = ['ApplyCol1', 'ApplyCol2']

myClass = BestPandas()
a = myClass.multiple_columns_shift(df.copy(), group_cols, apply_cols, shifts)
print(a.head(5))
p = ''