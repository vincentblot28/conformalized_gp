import io

import numpy as np
import pandas as pd
import scipy
from sklearn.datasets import fetch_california_housing

from mapie.utils import custom_scaler, get_std_from_triangular_law


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
    a1, b1 = 150, 200
    a2, b2 = 220, 300
    a3, b3 = 6, 10
    a4, b4 = -10, 10
    a5, b5 = 16, 45
    a6, b6 = .5, 1
    a7, b7 = .08, .18
    a8, b8 = 2.5, 6
    a9, b9 = 1700, 2500
    a10, b10 = .025, .08

    x1 = np.random.uniform(low=a1, high=b1, size=(nobs, 1))
    x2 = np.random.uniform(low=a2, high=b2, size=(nobs, 1))
    x3 = np.random.uniform(low=a3, high=b3, size=(nobs, 1))
    x4 = np.random.uniform(low=a4, high=b4, size=(nobs, 1)) * (np.pi/180)
    x5 = np.random.uniform(low=a5, high=b5, size=(nobs, 1))
    x6 = np.random.uniform(low=a6, high=b6, size=(nobs, 1))
    x7 = np.random.uniform(low=a7, high=b7, size=(nobs, 1))
    x8 = np.random.uniform(low=a8, high=b8, size=(nobs, 1))
    x9 = np.random.uniform(low=a9, high=b9, size=(nobs, 1))
    x10 = np.random.uniform(low=a10, high=b10, size=(nobs, 1))

    X = np.concatenate([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], axis=1)
    y = _wing_weight(X, noisy=noisy)
    mean = np.array(
        [(a1 + b1) / 2, (a2 + b2) / 2, (a3 + b3) / 2, (a4 + b4) / 2,
         (a5 + b5) / 2, (a6 + b6) / 2, (a7 + b7) / 2,
         (a8 + b8) / 2, (a9 + b9) / 2, (a10 + b10) / 2]
    )
    std = np.sqrt(
        np.array(
            [
                (b1 - a1)**2 / 12, (b2 - a2)**2 / 12, (b3 - a3)**2 / 12,
                ((b4 - a4) * np.pi / 180)**2 / 12, (b5 - a5)**2 / 12,
                (b6 - a6)**2 / 12, (b7 - a7)**2 / 12, (b8 - a8)**2 / 12,
                (b9 - a9)**2 / 12, (b10 - a10)**2 / 12
            ]
        )
    )
    X = custom_scaler(X, mean=mean, std=std)
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
    Z_est = np.random.multivariate_normal(np.repeat(0, d), cov, size=100000)
    X_est = scipy.stats.norm.cdf(Z_est)
    X = scipy.stats.norm.cdf(Z)
    mean = np.repeat(.5, d)
    std = np.std(X_est, axis=0)
    y = _noisy_morokoff(X, d, noisy)
    X = custom_scaler(X, mean=mean, std=std)
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

    mean = [101.6, 0.0233,  0.3, 0.05, 5.0*1e-6, 4.5*1e-9, 7.8*1e-4]
    std = [
        4.0, 0.0005,
        get_std_from_triangular_law(0.1, 0.3, 0.5),
        get_std_from_triangular_law(0.01, 0.05, 0.3),
        get_std_from_triangular_law(0.5*1e-6, 5.0*1e-6, 10.0*1e-6),
        get_std_from_triangular_law(1.0*1e-9, 4.5*1e-9, 8.0*1e-9),
        get_std_from_triangular_law(0.1*1e-4, 7.8*1e-4, 12*1e-4)
    ]
    X = custom_scaler(X, mean=mean, std=std)

    return X, y
