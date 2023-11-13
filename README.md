Conformalizing Gaussian Processes For More Robust Uncertainty Quantification
====================================================


**What is done in this repo** 

🔗 Requirements
===============
Python 3.7+ 

**OSX**: ACPI uses Cython extensions that need to be compiled with multi-threading support enabled. 
The default Apple Clang compiler does not support OpenMP.
To solve this issue, obtain the lastest gcc version with Homebrew that has multi-threading enabled: 
see for example [pysteps installation for OSX.](https://pypi.org/project/pysteps/1.0.0/)

**Windows**: Install MinGW (a Windows distribution of gcc) or Microsoft’s Visual C

Install the required packages:
- Via `pip`:

```
$ pip install -r requirements.txt
```

- Via conda:
```
$ conda install -f environment.yml
```

🛠 Installation
===============

Clone the repo and run the following command in the conformalized_gp directory to install the code
```
$ pip install .
```


⚡️ Quickstart
==============
Here is a quickstart to use the Jackknife+GP method on any regression dataset. Here, the goal is the compare
visually the results given by the standard Jackknife+ method, the Credibility Intervals and our methodology.


We first start to import the necessary packages
```python
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split

from mapie.conformity_scores.residual_conformity_scores import GPConformityScore
from mapie.regression import MapieRegressor

BLUE = np.array([[26, 54, 105]]) / 255
ORANGE = np.array([[223, 84, 49]]) / 255
YELLOW = np.array([[242, 188, 64]]) / 255
```

- In this example, we are going to work on an analytical function of our imagination which have some good visual behavior :

$`g(x) = 3x\sin(x) - 2x\cos(x) + \frac{x^3}{40} - \frac{x^2}{2} - 10x`$


```python
def g(x):
    return (3 * x * np.sin(x) - 2 * x * np.cos(x) + ( x ** 3) / 40 - .5 * x ** 2 - 10 * x)

x_mesh = np.linspace(-40, 60, 5000)
plt.plot(x_mesh, g(x_mesh))
plt.xlabel("$x$")
plt.ylabel("$g(x)$")
```
![toy function](https://github.com/vincentblot28/conformalized_gp/blob/ADD-readme-and-toy-notebook/plots/toy_function.png)

- Then we split our data into train and test and train au sickit-learn `GaussianProcessRegressor` with a `RBF` kernel.

```python 
X_train, X_test, y_train, y_test = train_test_split(x_mesh, g(x_mesh), test_size=.98, random_state=42)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
gp = GaussianProcessRegressor(normalize_y=True)
gp.fit(X_train, y_train)
```

- We then define and train the two conformal methods (J+ and J+GP):
```python 
mapie_j_plus_gp = MapieRegressor(
    estimator=gp,
    cv=-1,
    method="plus",
    conformity_score=GPConformityScore(),
    model_has_std=True,
    random_state=42
)

mapie_j_plus = MapieRegressor(
    estimator=gp,
    cv=-1,
    method="plus",
    conformity_score=None,
    model_has_std=False,
    random_state=42
)


mapie_j_plus_gp.fit(X_train, y_train)
mapie_j_plus.fit(X_train, y_train)
```

- Finally,  we predict and compute prediction intervals with a confidence level of 90% on the test set and plot the prediction intervals of the three methods

```python
ALPHA = .1

_, y_pss_j_plus_gp = mapie_j_plus_gp.predict(x_mesh.reshape(-1, 1), alpha=ALPHA)
_, y_pss_j_plus = mapie_j_plus.predict(x_mesh.reshape(-1, 1), alpha=ALPHA)

y_mean, y_std = gp.predict(x_mesh.reshape(-1, 1), return_std=True)

q_alpha_min = scipy.stats.norm.ppf(ALPHA / 2)
q_alpha_max = scipy.stats.norm.ppf(1 - ALPHA / 2)

f, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.scatter(X_train, y_train, c=BLUE)


ax.plot(x_mesh, g(x_mesh), c=BLUE)
ax.plot(x_mesh, y_mean, c=YELLOW)
ax.fill_between(
        x_mesh,
        y_mean + y_std * q_alpha_min,
        y_mean + y_std * q_alpha_max,
        alpha=0.3,
        color=YELLOW,
        label=r"$\pm$ 1 std. dev.",
    )


ax.fill_between(
        x_mesh,
        y_pss_j_plus_gp[:, 0, 0],
        y_pss_j_plus_gp[:, 1, 0],
        alpha=.6,
        color=ORANGE,
        label=r"$\pm$ 1 std. dev.",
    )

ax.fill_between(
        x_mesh,
        y_pss_j_plus[:, 0, 0],
        y_pss_j_plus[:, 1, 0],
        alpha=.3,
        color="g",
        label=r"$\pm$ 1 std. dev.",
    )
ax.legend(
    [
        "Training Points",
        "True function", "Mean of posterior GP",
        "Posterior GP Credibility Interval",
        "Prediction Interval J+GP",
         "Prediction Interval J+", 
    ]
)
ax.set_xlabel("$x$")
ax.set_ylabel("$g(x)$")
```
![toy function intervals](https://github.com/vincentblot28/conformalized_gp/blob/ADD-readme-and-toy-notebook/plots/intervals_toy_function.png)

