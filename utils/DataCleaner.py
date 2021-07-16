import pandas as pd
import numpy as np


# TODO : check score with between area and bedroomCount and without

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

        self.df['typeProperty'] = self.df['typeProperty'].apply(lambda x: 1 if x == 'HOUSE' else 0)

        # Dropping rows with price as NaN values
        self.df = self.df[self.df['price'].notna()]
        self.df = self.df[self.df['area'].notna()]

        self.df.terraceSurface.fillna(0, inplace=True)
        self.df.gardenSurface.fillna(0, inplace=True)
        self.df["outsideSpace"] = self.df["terraceSurface"] + self.df["gardenSurface"]

        self.df.facadeCount.fillna(0, inplace=True)

        # Dropping duplicated values
        self.df = self.df.drop_duplicates(subset=['area', 'price'], keep='last')

        # Deleting least corrleated columns
        self.df = self.df.drop(
            ['kitchenType', 'typeSale', 'subtypeSale', 'terraceSurface', 'isFurnished', 'gardenSurface'], axis=1)

        subtypeProperty = pd.DataFrame(self.df.groupby('subtypeProperty').sum().index.to_list())

        def getIndexSubtypeProp(x):
            return subtypeProperty.index[subtypeProperty[0] == x].tolist()[0]

        self.df['subtypeProperty'] = self.df['subtypeProperty'].apply(
            lambda x: getIndexSubtypeProp(x) if not pd.isnull(x) else -1)

        buildingCondition = pd.DataFrame(self.df.groupby('buildingCondition').sum().index.to_list())

        print(buildingCondition)

        def getBuildingCondition(x):
            return buildingCondition.index[buildingCondition[0] == x].tolist()[0]

        self.df['buildingCondition'] = self.df['buildingCondition'].apply(
            lambda x: getBuildingCondition(x) if not pd.isnull(x) else -1)
