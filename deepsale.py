"""
    Predicting boston housing market values with neural nets
"""
import numpy as np
import pandas as pd
from keras.layers.core import Activation, Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_train_test(bos_df: pd.DataFrame) -> tuple:
    """
        Obtain the training and testing data.
    """
    X_data = []
    Y_data = []

    for row in bos_df.values:
        X_data.append(row[:-1])
        Y_data.append(row[-1])

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, random_state=1)
    return np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)


def get_bos_df() -> pd.DataFrame:
    """
        Obtain the boston dataframe.
    """
    boston_data = load_boston()
    bos_df = pd.DataFrame(boston_data.data)
    bos_df.columns = boston_data.feature_names
    bos_df["PRICE"] = boston_data.target

    return bos_df


def simple_linear_regression():
    """
        Create our simple linear regression model.
    """
    # Obtain the boston df
    bos_df = get_bos_df()
    scaler = MinMaxScaler()

    # Obtain training and testing data (Apply scaling as well)
    X_train, X_test, Y_train, Y_test = get_train_test(bos_df)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Create the NN and train it
    model = Sequential()
    model.add(Dense(1, input_dim=13, activation="linear"))
    model.compile(optimizer="adam", loss="mse", metrics=["mse", "mae"])
    model.fit(X_train, Y_train, epochs=400, batch_size=1, verbose=1)

    # Create the LR and train it
    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    # Compare results of both models
    loss, mse, mae = model.evaluate(X_test, Y_test, verbose=0)
    sklearn_mse = mean_squared_error(lr.predict(X_test), Y_test)
    print(f"MSE for Keras NN: {mse}")
    print(f"MSE for LR model: {sklearn_mse}")
    print(f"r2 score for Keras NN: {r2_score(model.predict(X_test), Y_test)}")
    print(f"r2 score for LR model: {r2_score(lr.predict(X_test), Y_test)}")


def simple_logistic_regression():
    """
    """
    dataframe = pd.read_csv("diabetes.csv")
    scaler = MinMaxScaler()

    X_train, X_test, Y_train, Y_test = get_train_test(dataframe)

    pass


def complex_linear_regression():
    """
        Create a complex linear regression model.
    """

    bos_df = get_bos_df()
    scaler = MinMaxScaler()

    X_train, X_test, Y_train, Y_test = get_train_test(bos_df)

    model = Sequential()
    model.add(Dense(18, input_dim=13))
    model.add(Activation("relu"))
    model.add(Dense(9))
    model.add(Dropout(0.1))
    model.add(Activation("relu"))
    model.add(Dense(4))
    model.add(Dropout(0.05))
    model.add(Activation("relu"))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(optimizer="adam", loss="mse", metrics=["mse", "mae"])
    model.fit(
        scaler.fit_transform(X_train), Y_train, epochs=200, batch_size=1, verbose=1
    )
    loss, mse, mae = model.evaluate(scaler.fit_transform(X_test), Y_test, verbose=0)
    print(f"MSE = {mse}")
    print(f"MAE = {mae}")
    print(f"r^2 = {r2_score(model.predict(scaler.fit_transform(X_test)), Y_test)}")


def complex_logistic_regression():
    """
        Create our complex logistic regression model.
    """
    dataframe = pd.read_csv("diabetes.csv")
    scaler = MinMaxScaler()

    X_train, X_test, Y_train, Y_test = get_train_test(dataframe)

    model = Sequential()
    model.add(Dense(16, input_dim=8))
    model.add(Activation("sigmoid"))
    model.add(Dense(8))
    model.add(Dropout(0.1))
    model.add(Activation("sigmoid"))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print(np_utils.to_categorical(Y_train))
    model.fit(
        scaler.fit_transform(X_train),
        np_utils.to_categorical(Y_train),
        epochs=200,
        batch_size=1,
        verbose=0,
    )
    loss, accuracy = model.evaluate(
        scaler.fit_transform(X_test), np_utils.to_categorical(Y_test)
    )
    print(f"Accuracy = {accuracy}")


simple_linear_regression()
