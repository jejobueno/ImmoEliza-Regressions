import pandas as pd

from utils.DataCleaner import DataCleaner
from utils.DataRegressor import DataRegressor

# We set and unlimited max of columns to print on console for dataframes
pd.set_option('display.max_columns', None)

# We read our data recollected from Immobile
data = pd.read_csv('assets/housing-data.csv', index_col=0)

# Create a new object DataCleaner to clean our data
# and visualize it
dataAnalyser = DataCleaner()
df = dataAnalyser.clean(data, visualize_flag=True)

# Create a new object DataRegressor which will contain
# our trained model for future predictions with different dataframes
dataRegressor = DataRegressor(df)

# Train the model of the regressor
dataRegressor.trainModel()

# Suppose that our first 1000 rows of the same dataframe are a new dataframe
data = pd.read_csv('assets/housing-data.csv', index_col=0).head(1000)

# We cleaned as our ancient dataframe used to train the model in a new data cleaner
dataAnalyser = DataCleaner()
df = dataAnalyser.clean(data)

# We do a prediction with our trained model
predictions = dataRegressor.predict(df)