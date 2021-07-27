import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class DataRegressor:
    def __init__(self, df: pd.DataFrame):
        """
        It creates a DataRegressor object containing the dataFrame we are going
        to train it with. Also contains a regressor which will be trained for
        futures predictions
        :param df: cleaned dataframe to train our model
        """
        self.df = df
        self.regressor = LinearRegression()
        self.newData = pd.DataFrame()

    def rescale(self, df):
        """This static method will standardize some of the features,
        all the areas (square meter) will be rescaled into their square root
        and the price into its logarithm
        :param df: cleaned data frame to resale
        :return: rescaleded dataframe
        """

        df["price"] = np.log(df["price"])
        df["area"] = np.sqrt(df["area"])
        df["outsideSpace"] = np.sqrt(df["outsideSpace"])
        df["landSurface"] = np.sqrt(df["landSurface"])

        return df

    def trainModel(self):
        # We first rescale our price and surfaces to get a better
        # observation of the linear relationship between them and help
        # our model to do better predictions
        self.df = self.rescale(self.df)

        plt.figure()
        plt.scatter(self.df.area, self.df.price, color="b")
        plt.title("Rescaled sqrtArea vs logPrice")
        plt.xticks(rotation=40)
        plt.xlabel("sqrtArea")
        plt.ylabel("logPrice")
        plt.tight_layout()
        plt.show()
        plt.savefig("assets/Rescaled sqrtArea vs logPrice", transparent=True)

        # We split our target and our features in numpy arrays
        y = self.df["price"].to_numpy()
        X = self.df.drop(["price"], axis=1).to_numpy()

        # We split our data into a train and test datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.10, random_state=42
        )

        # Se fit our regression model to the train data
        self.regressor.fit(X_train, y_train)

        # print scores
        print("############# LINEAR REGRESSOR #############")
        print("Train score", self.regressor.score(X_train, y_train))
        print("Test score", self.regressor.score(X_test, y_test))

        # Make predictions on test datasets
        predictions = self.regressor.predict(X_test)

        # Print the results
        plt.figure()
        plt.scatter(y_test, predictions, color="b")
        plt.title("predictions VS y_test")
        plt.xlabel("y_test")
        plt.ylabel("predictions")
        plt.savefig("assets/predictions VS y_test.png", transparent=True)

        plt.figure()
        plt.scatter(X_test[:, 1] ** 2, np.exp(y_test), color="r")
        plt.scatter(X_test[:, 1] ** 2, np.exp(predictions), color="b")
        plt.title("predictions vs original data")
        plt.legend(["predictions", "original data"])
        plt.xlabel("area")
        plt.ylabel("price")
        plt.savefig(
            "assets/predictions vs original data.png", transparent=True
        )
        plt.show()

    def predict(self, df):
        """
        This method receives a new dataframe to make predictions with the
        regressor already trained
        :param df: cleaned dataframe ready to create predictions:
        :return: predictions for price
        """
        self.adjustToTrainedModelDF(df)
        self.newData = self.rescale(self.newData)

        y = self.newData["price"].to_numpy()
        X = self.newData.drop(["price"], axis=1).to_numpy()

        print("############# LINEAR REGRESSOR FOR NEW DATA #############")
        print("score", self.regressor.score(X, y))
        predictions = self.regressor.predict(X)

        plt.figure()
        plt.scatter(y, predictions, color="r")
        plt.title("predictions VS y")
        plt.xlabel("y_test")
        plt.ylabel("predictions")
        plt.savefig("assets/predictions VS y.png")

        plt.figure()
        plt.scatter(X[:, 1] ** 2, np.exp(predictions), color="b")
        plt.scatter(X[:, 1] ** 2, np.exp(y), color="r")
        plt.title("predictions vs data")
        plt.legend(["predictions", "original data"])
        plt.xlabel("area")
        plt.ylabel("price")
        plt.tight_layout()
        plt.savefig("assets/predictions vs data.png")
        plt.show()

        return np.exp(self.regressor.predict(X))

    def adjustToTrainedModelDF(self, df):
        """
        This method fit the new data into the format of the X dataset from
         the model to be able to predict using the model of the regressor
        :param df: new dataframe to fit into the X format.
        :return: None
        """
        # We create a new data frame with the columns of the dataframe used to
        # train the model
        self.newData = pd.DataFrame(columns=self.df.columns.to_list())

        # We append our new data to this dataframe
        self.newData = self.newData.append(df)

        # Fill all the nan values with zeros
        self.newData.fillna(0, inplace=True)
