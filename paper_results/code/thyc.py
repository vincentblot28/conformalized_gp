from copy import copy

import scipy
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mapie.conformity_scores.residual_conformity_scores import (
    GPConformityScore
)
from mapie.metrics import (
    regression_coverage_score_v2,
    regression_mean_width_score
)
from mapie.regression import MapieRegressor
from mapie.metrics import spearman_correlation, q2
from datasets import get_thyc
from wrappers import GpOTtoSklearnStd


X, y = get_thyc()


print(X.shape, y.shape)

# Split into train test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42
)

print(
    f"  N training points = {len(X_train)}\n",
    f" N testing points = {len(X_test)}"
)


# Get data


scaler = StandardScaler().fit(X_train)
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)


# Define all possible models

noise = None


nus = [1/2, 3/2, 5/2]

models_hp = {
    "GP": {
        "nu": {nu: [1] for nu in nus}
    },
    "J+": {
        "nu": {nu: [1] for nu in nus}
    },
    "J+GP": {
        "nu": {nu: [.5, 1, 1.5] for nu in nus}
    },
    "J-minmax-GP": {
        "nu": {nu: [.5, 1, 1.5] for nu in nus}
    }
}


models = {}

for method in models_hp.keys():
    for nu in models_hp[method]["nu"].keys():
        for power_std in models_hp[method]["nu"][nu]:
            models[(method, nu, power_std)] = {
                "estimator": GpOTtoSklearnStd(
                    scale=1, amplitude=1.0, nu=nu, noise=noise
                )
            }

global_models = {}
for nu in nus:
    global_models[nu] = {
        "estimator": GpOTtoSklearnStd(
            scale=1, amplitude=1.0, nu=nu, noise=noise
        )
    }


models.keys()


for model_name, model in models.items():
    if model_name[0] == "GP":
        print("Fitting ", model_name)
        model["estimator"].fit(X_train_scale, y_train)


for model_name, model in models.items():
    if model_name[0] == "GP":
        print(
            model_name,
            "MSE:",
            mean_squared_error(
                y_test, model["estimator"].predict(X_test_scale)
            )
        )


for model_name, model in models.items():
    if model_name[0] == "GP":
        print(
            model_name,
            "Q2:",
            q2(
                y_test, model["estimator"].predict(X_test_scale)
            )
        )


# Fit MAPIE


for model_name, model in global_models.items():
    global_models[model_name]["mapie_estimator_std"] = MapieRegressor(
        estimator=model["estimator"],
        conformity_score=GPConformityScore(sym=True),
        cv=-1,
        method="plus",
        model_has_std=True,
        random_state=42
    )
    global_models[model_name]["mapie_estimator_no_std"] = MapieRegressor(
        estimator=model["estimator"],
        conformity_score=None,
        cv=-1,
        method="plus",
        model_has_std=False,
        random_state=42
    )

for model_name, model in global_models.items():
    print("Fitting Global model std", model_name)
    model["mapie_estimator_std"].fit(X_train_scale, y_train)
    print("Fitting Global model no std", model_name)
    model["mapie_estimator_no_std"].fit(X_train_scale, y_train)

for model_name, model in models.items():
    if model_name[0] != "GP":
        nu = model_name[1]
        method = "plus" if "+" in model_name[0] else "minmax"
        if "GP" in model_name[0]:
            cs = GPConformityScore(sym=True, power=model_name[2])
            estimator_ = copy(
                global_models[nu]["mapie_estimator_std"].estimator_
            )
            estimator_.method = method
        else:
            cs = None
            estimator_ = copy(
                global_models[nu]["mapie_estimator_no_std"].estimator_
            )
            estimator_.method = "plus"

        models[model_name]["mapie_estimator"] = MapieRegressor(
            estimator=estimator_,
            conformity_score=cs,
            cv=-1,
            method=method,
            model_has_std=True if "GP" in model_name[0] else False,
            random_state=42
        )

for model_name, model in models.items():
    if model_name[0] != "GP":
        print(model_name)
        print(model["mapie_estimator"].method)
        print(model["mapie_estimator"].estimator.method)


for model_name, model in models.items():
    if model_name[0] != "GP":
        print("Fitting MAPIE", model_name)
        model["mapie_estimator"].fit(X_train_scale, y_train)

# Coverage
ALPHA = np.array([.1, .05, .01])
q_alpha_min = scipy.stats.norm.ppf(ALPHA / 2)
q_alpha_max = scipy.stats.norm.ppf(1 - ALPHA / 2)
for model_name, model in models.items():
    if model_name[0] == "GP":
        y_mean, y_std = model["estimator"].predict(
            X_test_scale, return_std=True
        )
        y_pss_gp = np.concatenate(
            [
                (
                    y_mean.reshape(-1, 1) +
                    y_std.reshape(-1, 1) * q_alpha_min.reshape(1, -1)
                )[:, np.newaxis, :],
                (
                    y_mean.reshape(-1, 1) +
                    y_std.reshape(-1, 1) * q_alpha_max.reshape(1, -1)
                )[:, np.newaxis, :]
            ],
            axis=1
        )
        model["y_pss"] = y_pss_gp


for model_name, model in models.items():
    if model_name[0] != "GP":
        print("Predict MAPIE", model_name)
        _, y_pss = model["mapie_estimator"].predict(X_test_scale, alpha=ALPHA)
        model["y_pss"] = y_pss


for model_name, model in models.items():
    model["coverage"] = [
        regression_coverage_score_v2(y_test, model["y_pss"][:, :, i])
        for i, _ in enumerate(ALPHA)
    ]
    model["average_width"] = [
        regression_mean_width_score(
            model["y_pss"][:, 1, i], model["y_pss"][:, 0, i]
        )
        for i, _ in enumerate(ALPHA)
    ]

# Correlation between width of the Prediction Interval and the model error


for model_name, model in models.items():
    if model_name[0] != "GP":
        model["errors"] = np.abs(
            model["mapie_estimator"].predict(X_test_scale, alpha=None) - y_test
        )
        model["width"] = np.abs(
            model["y_pss"][:, 0, :] - model["y_pss"][:, 1, :]
        )
    else:
        model["errors"] = np.abs(
            model["estimator"].predict(X_test_scale) - y_test
        )
        model["width"] = np.abs(
            model["y_pss"][:, 0, :] - model["y_pss"][:, 1, :]
        )

for model_name, model in models.items():
    model["spearman_correlation_to_error"] = []
    for index_confidence in range(len(ALPHA)):
        str_vect = [
            e + '--' + w for e, w in zip(
                model["errors"].astype("str").tolist(),
                model["width"][:, index_confidence].astype("str").tolist()
            )
        ]
        model["spearman_correlation_to_error"].append(bootstrap(
            (np.array(str_vect), ),
            spearman_correlation,
            axis=0,
            n_resamples=999
        ))


# Get average width of the prediction intervals for each model in a pandas DF
index = pd.MultiIndex.from_tuples([("Average width", i) for i in 1 - ALPHA])
df_width = pd.DataFrame(
    {
        model_name: model["average_width"]
        for model_name, model in models.items()
    },
    index=index
)


index = pd.MultiIndex.from_tuples([("Coverage", i) for i in 1 - ALPHA])

df_cov = pd.DataFrame(
    {
        model_name: [c[0] for c in model["coverage"]]
        for model_name, model in models.items()
    },
    index=index
)


index = pd.MultiIndex.from_tuples(
    [("Spearman Correlation", i) for i in 1 - ALPHA]
)

df_spearman = pd.DataFrame(
    {
        model_name: [
            np.mean(c.bootstrap_distribution) for c in model[
                "spearman_correlation_to_error"
            ]
        ]
        for model_name, model in models.items()
    },
    index=index
)

df_results = pd.concat([df_cov.T, df_width.T, df_spearman.T], axis=1)

df_results.to_csv("paper_results/table_results/thyc_results.csv")
