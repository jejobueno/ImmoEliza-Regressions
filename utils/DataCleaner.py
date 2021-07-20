import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataCleaner:
    def __init__(self):
        self.df = pd.DataFrame

    def clean(self, df: pd.DataFrame, visualize_flag: bool = False) -> pd.DataFrame:
        """
        This method will clean the dataframe deleting duplicated rows, fixing erros,
        eliminating outliers. If visualize_flag set will plot correlation map
        and relationships between the variables and the target price
        :param df: dataframe to clean
        :param visualize_flag: Will plot all relationships with variable price
        :return: cleaned dataframe
        """
        self.df = df

        # We clean first all the entirely empty rows
        self.df.dropna(how='all', inplace=True)

        # We delete the blank spaces at the beginning and end of each string
        self.df.apply(lambda x: x.strip() if type(x) == str else x)

        # Fixing errors
        # Fixing variable hasFullyEquippedKitchen
        has_hyperEquipped = self.df['kitchenType'].apply(lambda x: 1 if x == 'HYPER_EQUIPPED' else 0)
        has_USHyperEquipped = self.df['kitchenType'].apply(lambda x: 1 if x == 'USA_HYPER_EQUIPPED' else 0)
        self.df.hasFullyEquippedKitchen = has_hyperEquipped | has_USHyperEquipped

        # Deleting variable 'typeProperty', keeping 'subtypeProperty'
        del self.df['typeProperty']

        # Dropping rows with price as NaN values
        self.df = self.df[self.df['price'].notna()]
        self.df = self.df[self.df['area'].notna()]

        # Dropping duplicated values
        self.df = self.df.drop_duplicates(subset=['area', 'price'], keep='last')

        # cleaning features with less than 5 occurrences
        features = ['postalCode', 'facadeCount', 'subtypeProperty', 'BedroomsCount']
        for feature in features:
            self.df = self.df[self.df[feature].map(self.df[feature].value_counts()) > 5]

        # Visualize relations between the variables throughout plots
        if visualize_flag:
            self.visualize()

        # Dropping outliers
        self.df = self.df[self.df['price'] < 6000000]
        self.df = self.df[self.df['area'] < 1350]

        # Creating new variable adding outside surface
        self.df.terraceSurface.fillna(0, inplace=True)
        self.df.gardenSurface.fillna(0, inplace=True)
        self.df["outsideSpace"] = self.df["terraceSurface"] + self.df["gardenSurface"]

        # Filling nan values to 0 for facadeCount
        self.df.facadeCount.fillna(0, inplace=True)

        # Deleting least correlated columns
        self.df = self.df.drop(
            ['kitchenType', 'typeSale', 'subtypeSale', 'terraceSurface', 'isFurnished', 'gardenSurface'], axis=1)

        # Transform  variables into features
        features = ['postalCode', 'buildingCondition', 'subtypeProperty', 'fireplaceExists',
                    'hasSwimmingPool', 'hasGarden', 'hasTerrace', 'hasFullyEquippedKitchen']
        for feature in features:
            cv_dummies = pd.get_dummies(self.df[feature])
            if cv_dummies.columns.__len__() < 3:
                cv_dummies.columns = [feature + 'True', feature + 'False']
            self.df = pd.concat([self.df, cv_dummies], axis=1)
            del self.df[feature]

        return self.df

    def visualize(self):
        """
        THis method will plot the heatmap correlation and the scatter plot between the
        different features and the target price from
        :return: None
        """

        # plotting the correlation heatmap
        plt.figure()
        features = self.df.drop('postalCode', axis=1)
        sns.heatmap(features.corr(), center=0, cmap="YlGnBu")
        plt.tight_layout()
        plt.xticks(rotation=40)
        plt.show()
        plt.savefig('assets/Correlation map.png')

        # Plotting all variables respect to price
        for feature in self.df.columns:
            if feature != 'price':
                plt.figure()
                sns.scatterplot(x=feature, y='price', data=self.df)
                plt.title(feature + ' vs price')
                plt.xlabel(feature)
                plt.ylabel('price')
                plt.xticks(rotation=40)
                plt.savefig('assets/' + feature + ' vs price.png')
                plt.tight_layout()
                plt.show()
