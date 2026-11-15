import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def read_csv(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file and returns its content as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file is empty.") from exc
    if data.empty:
        raise ValueError("The CSV file is empty.")
    required_columns = {"Size", "Price"}
    if not required_columns.issubset(set(data.columns)):
        raise ValueError("CSV file must contain 'Size' and 'Price' columns.")
    return data


def train_polynomial_regression(X: np.array, y: np.array, degree: int):
    """
    Trains a polynomial regression model.
    """
    X_arr = np.asarray(X).reshape(-1, 1)
    y_arr = np.asarray(y)

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_arr)

    model = LinearRegression()
    model.fit(X_poly, y_arr)
    return model, poly


def predict_prices(model: LinearRegression, poly: PolynomialFeatures, X: np.array):
    """
    Predicts prices using the trained polynomial regression model.
    """
    X_arr = np.asarray(X).reshape(-1, 1)
    X_poly = poly.transform(X_arr)
    return model.predict(X_poly)


def evaluate_model(y_true: np.array, y_pred: np.array):
    """
    Evaluates the model using various metrics.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


def visualize_data(X: np.array, y: np.array, model: LinearRegression, poly: PolynomialFeatures):
    """
    Visualizes the original data and the regression curve.
    """
    X_arr = np.asarray(X)
    X_plot = np.linspace(X_arr.min(), X_arr.max(), 100)
    y_plot = predict_prices(model, poly, X_plot)

    plt.scatter(X_arr, y, color="blue", label="Actual Prices")
    plt.plot(X_plot, y_plot, color="red", label="Polynomial Regression Fit")
    plt.xlabel("House Size")
    plt.ylabel("House Price")
    plt.title("Polynomial Regression: House Size vs Price")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    file_path = "../house_data.csv"
    data = read_csv(file_path)

    X = data["Size"].values
    y = data["Price"].values

    degree = 3
    model, poly = train_polynomial_regression(X, y, degree)
    predictions = predict_prices(model, poly, X)

    metrics = evaluate_model(y, predictions)
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

    visualize_data(X, y, model, poly)
