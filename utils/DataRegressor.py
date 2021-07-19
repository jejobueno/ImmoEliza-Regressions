import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import ensemble


class DataRegressor:

    def __init__(self, df):
        self.df = df
        self.regressor = LinearRegression()
        self.newData = pd.DataFrame()

    @staticmethod
    def standardize(df):
        df['price'] = np.log(df['price'])
        df['area'] = np.sqrt(df['area'])
        df['outsideSpace'] = np.sqrt(df['outsideSpace'])
        df['landSurface'] = np.sqrt(df['landSurface'])

        plt.figure()
        plt.scatter(df.area, df.price)
        plt.title('normalized data with log and sqrt')
        plt.show()

        return df

    def trainModel(self):
        self.df = self.standardize(self.df)

        y = self.df['price'].to_numpy()
        X = self.df.drop(['price'], axis=1).to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

        self.regressor.fit(X_train, y_train)
        print( '############# LINEAR REGRESSOR #############')
        print('Train score', self.regressor.score(X_train, y_train))
        print('Test score', self.regressor.score(X_test, y_test))
        predictions = self.regressor.predict(X_test)

        plt.figure()
        plt.scatter(y_test, predictions, color='r')
        plt.title('Degree ' + str(self.regressor))
        plt.xlabel('y_test')
        plt.ylabel('predictions')

        plt.figure()
        plt.scatter(X_test[:,1]**2, np.exp(predictions), color='b')
        plt.scatter(X_test[:,1]**2, np.exp(y_test), color='r')
        plt.xlabel('area')
        plt.ylabel('price')
        """
        # GRADIENT BOOST
        clf = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2,
                                                 learning_rate=0.1, loss='ls')
        clf.fit(X_train, y_train)
        print( '############# BOOST #############')
        print('Train score', clf.score(X_train, y_train))
        print('Test score', clf.score(X_test, y_test))

        predictions = clf.predict(X_test)
        plt.scatter(y_test, predictions, color='r')
        plt.title('Degree ' + str(degree))
        plt.xlabel('y_test')
        plt.ylabel('predictions boosted')

        plt.figure()
        plt.scatter(X_test[:,2]**2, np.exp(predictions), color='b')
        plt.scatter(X_test[:,2]**2, np.exp(y_test), color='r')
        plt.title('PREDICTIONS BOOSTED')
        plt.xlabel('area')
        plt.ylabel('price')
        """

    def predict(self, df):
        self.adjustToTrainedModelDF(df)
        self.newData = self.standardize(self.newData)

        y = self.newData['price'].to_numpy()
        X = self.newData.drop(['price'], axis=1).to_numpy()

        return np.exp(self.regressor.predict(X))

    def adjustToTrainedModelDF(self, df):
        self.newData = pd.DataFrame(columns=self.df.columns.to_list())
        self.newData = self.newData.append(df)
        self.newData.fillna(0, inplace=True)



