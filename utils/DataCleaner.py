import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

        # Dropping outliers
        self.df = self.df[self.df['price'] < 7000000]
        self.df = self.df[self.df['area'] < 1350]

        # ploting behaviors
        #plt.scatter(self.df.area, self.df.price)
        #plt.show()

        # Deleting least correlated columns
        self.df = self.df.drop(
            ['kitchenType', 'typeSale', 'subtypeSale', 'terraceSurface', 'isFurnished', 'gardenSurface'], axis=1)

        # cleaning features with less than 5 occurrences
        features = ['postalCode', 'facadeCount', 'subtypeProperty', 'BedroomsCount']
        for feature in features:
            self.df = self.df[self.df[feature].map(self.df[feature].value_counts()) > 5]

        # Transform  variables into features
        features = ['postalCode', 'buildingCondition', 'subtypeProperty', 'fireplaceExists',
                    'hasSwimmingPool', 'hasGarden', 'hasTerrace', 'hasFullyEquippedKitchen']
        for feature in features:
            cv_dummies = pd.get_dummies(self.df[feature])
            if cv_dummies.columns.__len__() < 3:
                cv_dummies.columns = [feature + 'True', feature + 'False']
            self.df = pd.concat([self.df, cv_dummies], axis=1)
            del self.df[feature]

