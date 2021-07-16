import pandas as pd
import numpy as np


class DataCleaner:
    def __init__(self, df):
        self.df = df
        self.housing_data = dict()


    def clean(self) -> pd.DataFrame:
        # We clean first all the entirely empty rows
        self.df.dropna(how='all', inplace=True)

        # We delete the blank spaces at the beginning and end of each string
        self.df.apply(lambda x: x.strip() if type(x) == str else x)

        # Fixing errors
        # Fixing variable hasFullyEquippedKitchen
        has_hyperEquipped = self.df['kitchenType'].apply(lambda x: 1 if x == 'HYPER_EQUIPPED' else 0)
        has_USHyperEquipped = self.df['kitchenType'].apply(lambda x: 1 if x == 'USA_HYPER_EQUIPPED' else 0)
        self.df.hasFullyEquippedKitchen = has_hyperEquipped | has_USHyperEquipped

        # Dropping rows with price as NaN values
        self.df = self.df[self.df['price'].notna()]
        self.df = self.df[self.df['area'].notna()]

        # Dropping duplicated values
        self.df = self.df.drop_duplicates(subset=['area', 'price'], keep='last')

        # Deleting constant variable 'typeSale'
        del self.df['typeSale']

