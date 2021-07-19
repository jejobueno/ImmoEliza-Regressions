import pandas as pd

from utils.DataCleaner import DataCleaner
from utils.DataRegressor import DataRegressor

data = pd.read_csv('/Users/pauwel/Documents/GitHub/ImmoEliza-Regressions/assets/housing-data.csv', index_col=0)

dataAnalyser = DataCleaner(data)
dataAnalyser.clean()

dataRegressor = DataRegressor(dataAnalyser.df)
dataRegressor.predict()