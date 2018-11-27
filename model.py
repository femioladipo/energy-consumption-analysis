from statsmodels.api import tsa
import pandas as pd


class EnergyModel:
    def __init__(self):
        self.n_training = None
        self.model = None

    def preprocess_training_data(self, df):
        df.set_index(pd.to_datetime(df.day), inplace=True)
        df.drop("day", axis=1, inplace=True)
        return df, None

    def fit(self, X, y):
        print(y)
        self.n_training = X.shape[0]
        # ar = tsa.AR(X)
        # opt_lag = ar.select_order(10, ic="aic")
        self.model = tsa.ARMA(X, order=(9, 5)).fit()

    def preprocess_unseen_data(self, df):
        df.set_index(pd.to_datetime(df.day), inplace=True)
        df.drop("day", axis=1, inplace=True)
        return df

    def predict(self, X):
        return self.model.predict(start=self.n_training,
                                  end=self.n_training + X.shape[0] - 1)


# arma = tsa.ARMA(time_series, order=(3, 3))
# arma_result = arma.fit()
# prediction = arma_result.predict(start=3)
