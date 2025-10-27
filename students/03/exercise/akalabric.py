import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

FEATURE = "advertising_impressions"
TARGET = "website_visits"


def main(predict_impressions=None, csv_path=None, save_plot=False):
    csv_path = csv_path or "students/03/data/20_website_visits.csv"
    df = pd.read_csv(csv_path)

    for col in (FEATURE, TARGET):
        if col not in df.columns:
            raise ValueError(f"Nedostaje stupac: {col}")
    df = df[[FEATURE, TARGET]].dropna()

    X = df[[FEATURE]] 
    y = df[TARGET]

    model = LinearRegression().fit(X, y)

    y_hat = model.predict(X)
    r2 = r2_score(y, y_hat)

    print("Koeficijent (slope):", float(model.coef_[0]))
    print("Presjek (intercept):", float(model.intercept_))
    print("R^2:", round(r2, 4))
    print(f"Model:{TARGET}={model.coef_[0]:.6f}*{FEATURE}+{model.intercept_:.6f}")

    plt.scatter(X[FEATURE], y, label="Podaci")

    x_min = X[FEATURE].min()
    x_max = X[FEATURE].max()
    x_line = pd.DataFrame({FEATURE: np.linspace(x_min, x_max, 100)})
    y_line = model.predict(x_line)
    plt.plot(x_line[FEATURE], y_line, label="Linearni model")

    plt.xlabel("Advertising impressions")
    plt.ylabel("Website visits")
    plt.title("Website visits ~ Advertising impressions")
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plt.savefig("regression_plot.png", dpi=150)
        print("Graf spremljen u: regression_plot.png")
    else:
        plt.show()

    if predict_impressions is not None:
        X_new = pd.DataFrame({FEATURE: [float(predict_impressions)]})
        pred = model.predict(X_new)[0]
        print(f"Predviđen broj posjeta za {predict_impressions:.0f} impresija: {pred:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", type=float, default=None)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--save-plot", action="store_true")
    args = parser.parse_args()
    main(args.predict, args.csv, args.save_plot)
