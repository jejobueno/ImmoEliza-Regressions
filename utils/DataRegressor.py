import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


class DataRegressor:

    def __init__(self, df):
        self.df = df

    def standardize(self):

        self.df['price'] = np.log(self.df['price'])
        self.df['area'] = np.sqrt(self.df['area'])

        plt.scatter(self.df.area, self.df.price)
        plt.title('normalized data with log and sqrt')
        plt.draw()
        plt.show()

    def predict(self):
        self.standardize()

        y = self.df['price'].to_numpy()
        X = self.df.drop(['price'], axis=1).to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        
        degree = 1
        polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        polyreg.fit(X_train, y_train)
        print(polyreg.score(X_train, y_train))

        predictions = polyreg.predict(X_test)

        plt.figure()
        plt.scatter(y_test, predictions, color='r')
        plt.xlabel('y_test')
        plt.ylabel('predictions')
        plt.draw()
        plt.show()

        plt.figure()
        plt.scatter(X_test[:,2]**2, np.exp(predictions), color='b')
        plt.scatter(X_test[:, 2] ** 2, np.exp(y_test), color='r')
        plt.xlabel('X_test')
        plt.ylabel('predictions')
        plt.draw()
        plt.show()