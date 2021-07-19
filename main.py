import pandas as pd

from utils.DataCleaner import DataCleaner
from utils.DataRegressor import DataRegressor

pd.set_option('display.max_columns', None)

data = pd.read_csv('assets/housing-data.csv', index_col=0)

dataAnalyser = DataCleaner(data)
dataAnalyser.clean()

dataRegressor = DataRegressor(dataAnalyser.df)
dataRegressor.trainModel()

data = pd.read_csv('assets/housing-data.csv', index_col=0).head(1000)

dataAnalyser = DataCleaner(data)
dataAnalyser.clean()

predictions = dataRegressor.predict(dataAnalyser.df)
print('######## data to predict ############')
print(dataAnalyser.df.head().price)
print('#########" predictions ###########"')
print(predictions)
