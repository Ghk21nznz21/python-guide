''' Pretty way of displaying a Model's performance using PrettyTable '''
import pandas as pd
from prettytable import PrettyTable

# Create a DataFrame
fake_df = pd.DataFrame({
    "Model Name": ["Model 1", "Model 2", "Model 3"],
    "Accuracy": [0.85, 0.90, 0.88]
})

# Convert the DataFrame to a PrettyTable
table = PrettyTable()

# Set the column names
table.field_names = fake_df.columns.tolist()

# Set the rows 
table.rows = fake_df.values.tolist()