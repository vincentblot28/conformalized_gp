import io

import numpy as np
import pandas as pd
import scipy
from sklearn.datasets import fetch_california_housing


def get_mpg():
    with open("data/auto-mpg.data-original", "r") as file:
        data = file.read()

    # Replace any multiple spaces with a single space and split the lines
    data = "\n".join(" ".join(line.split()) for line in data.split("\n"))

    # Use pandas to read the data into a DataFrame
    column_names = [
        "mpg", "cylinders", "displacement", "horsepower",
        "weight", "acceleration", "year", "origin", "name"
    ]
    df = pd.read_csv(
        io.StringIO(data), header=None,
        delimiter=r"\s+", names=column_names
    ).dropna()

    y = df["mpg"].values
    X = df[[col for col in df.columns if col not in ["mpg", "name"]]].values

    return X, y


def _wing_weight(x, noisy=False):
    t1 = .036 * x[:, 0]**(.758)
    t2 = x[:, 1]**(.0035)
    t3 = (x[:, 2]/(np.cos(x[:, 3])**2))**(.6)
    t4 = x[:, 4]**(.006)
    t5 = x[:, 5]**(.04)
    t6 = ((np.cos(x[:, 3])) / (100 * x[:, 6]))**(.3)
    t7 = (x[:, 7] * x[:, 8])**(.49)
    t8 = x[:, 0] * x[:, 9]
    if noisy:
        noise = np.random.normal(0, 5, x.shape[0])
        return t1 * t2 * t3 * t4 * t5 * t6 * t7 + t8 + noise
    else:
        return t1 * t2 * t3 * t4 * t5 * t6 * t7 + t8


def get_wing_weight(noisy=False):
    nobs = 600
    np.random.seed(42)
    x1 = np.random.uniform(low=150, high=200, size=(nobs, 1))
    x2 = np.random.uniform(low=220, high=300, size=(nobs, 1))
    x3 = np.random.uniform(low=6, high=10, size=(nobs, 1))
    x4 = np.random.uniform(low=-10, high=10, size=(nobs, 1)) * (np.pi/180)
    x5 = np.random.uniform(low=16, high=45, size=(nobs, 1))
    x6 = np.random.uniform(low=.5, high=1, size=(nobs, 1))
    x7 = np.random.uniform(low=.08, high=.18, size=(nobs, 1))
    x8 = np.random.uniform(low=2.5, high=6, size=(nobs, 1))
    x9 = np.random.uniform(low=1700, high=2500, size=(nobs, 1))
    x10 = np.random.uniform(low=0.025, high=.08, size=(nobs, 1))
    X = np.concatenate([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], axis=1)
    y = _wing_weight(X, noisy=noisy)

    return X, y


def _noisy_morokoff(x, d, noisy):
    noise = np.random.normal(0, np.sqrt(1e-4), x.shape[0]) if noisy else 0
    return .5 * (1 + 1 / d)**d * (x ** (1 / d)).prod(axis=1) + noise


def get_morokoff(noisy=False, nobs=600):
    cov = np.array(
        [
            [1, .9, 0, 0, 0, .05, -.3, 0, 0, 0],
            [.9, 1, 0, 0, 0, 0, 0, .1, 0, 0],
            [0, 0, 1, 0, -.3, .1, .4, 0, .05, 0],
            [0, 0, 0, 1, .4, 0, 0, -.35, 0, 0],
            [0, 0, -.3, .4, 1, 0, 0, 0, .1, 0],
            [.05, 0, .1, 0, 0, 1, 0, 0, 0, 0],
            [-.3, 0, .4, 0, 0, 0, 1, 0, 0, -.3],
            [0, .1, 0, -.35, 0, 0, 0, 1, 0, 0],
            [0, 0, .05, 0, .1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, -.3, 0, 0, 1]
        ]
    )

    d = cov.shape[0]
    np.random.seed(42)
    Z = np.random.multivariate_normal(np.repeat(0, d), cov, size=nobs)
    X = scipy.stats.norm.cdf(Z)
    y = _noisy_morokoff(X, d, noisy)
    return X, y


def get_california(n="all"):
    data = fetch_california_housing()
    X = data["data"]
    y = data["target"]

    if n == "all":
        return X, y
    elif isinstance(n, int):
        np.random.seed(42)
        indices = np.random.choice(len(X), size=n, replace=False)
        return X[indices], y[indices]
    elif isinstance(n, float):
        if n <= 1:
            np.random.seed(40)
            indices = np.random.choice(
                len(X), size=int(n * len(X)), replace=False
            )
            return X[indices], y[indices]
        else:
            raise ValueError(
                "If n is a float, it should be less or equal to 1."
            )
    else:
        raise ValueError("n must be either equal to 'all', an int or a float")


def get_cpu():
    with open("data/machine.data") as f:
        data = f.readlines()
    data_clean = [line.split(",")[2:] for line in data]
    data_clean = np.array([[float(x) for x in line] for line in data_clean])
    return data_clean[:, :-1], data_clean[:, -1]


def get_thyc():
    X = np.load("data/thyc/x.npy")
    y = np.load("data/thyc/g_x.npy")

    return X, y
