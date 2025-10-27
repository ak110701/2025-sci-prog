import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def main(predict_impressions: float | None = None):
    df = pd.read_csv("students/03/data/20_website_visits.csv")

    X = df[["advertising_impressions"]]  
    y = df["website_visits"]             

    model = LinearRegression()
    model.fit(X, y)

    y_hat = model.predict(X)
    r2 = r2_score(y, y_hat)

    print("Koeficijent (slope):", float(model.coef_[0]))
    print("Presjek (intercept):", float(model.intercept_))
    print("R^2:", round(r2, 4))

    plt.scatter(X, y, label="Podaci")
    x_line = np.linspace(X.min()[0], X.max()[0], 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, label="Linearni model")
    plt.xlabel("Advertising impressions")
    plt.ylabel("Website visits")
    plt.title("Website visits ~ Advertising impressions")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if predict_impressions is not None:
        pred = model.predict(np.array([[float(predict_impressions)]]))[0]
        print(f"Predviđen broj posjeta za {predict_impressions:.0f} impresija: {pred:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", type=float, default=None,
                        help="Ako proslijediš broj impresija (npr. --predict 10000), ispisat ću predviđene posjete.")
    args = parser.parse_args()
    main(args.predict)
