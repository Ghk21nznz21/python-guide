''' Efficient way to convert large size Excel to Csv  '''

import os 
import time 
from xlsx2csv import Xlsx2csv
from io import StringIO
import pandas as pd
from joblib import Parallel, delayed


def read_excel(path: str) -> pd.DataFrame:
    # Convert the Excel file to CSV format and write it to the buffer
    buffer = StringIO()
    Xlsx2csv(f'Data/{path}', outputencoding="utf-8", sheet_name='Dados').convert(buffer)
    # Reset the buffer's position to the beginning
    buffer.seek(0)
    
    # Read the CSV data from the buffer 
    return pd.read_csv(buffer)

def merge():
    # Record the start time
    start = time.time()
    
    # Read all Excel files in parallel
    df_tickets = Parallel(n_jobs=-1, verbose=10)(
        delayed(read_excel)(path)
        for path in os.listdir('Data')
        if path.startswith('ML')
    )
    i = 0
    srd_tickets = None
    for tickets in df_tickets:
        if i == 0:
            srd_tickets = tickets
            i = 1 
        else:
            srd_tickets = pd.concat([srd_tickets, tickets])
    print(time.time()- start)
    srd_tickets.to_csv('Data/Tickets.csv', sep=',', index=False)


merge()
