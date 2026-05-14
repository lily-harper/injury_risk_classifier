import pandas as pd 
import numpy as np

def create_outcome(df):
    df = df.copy()
    df['injured'] = (df["seriously_injured"] > 0) | (df["fatalities"] > 0)
    df['injured'] = df['injured'].astype(int)

    return df 