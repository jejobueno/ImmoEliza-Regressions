import pandas as pd

from utils.DataCleaner import DataCleaner

data = pd.read_csv('assets/housing-data.csv', index_col=0)

dataAnalyser = DataCleaner(data)

