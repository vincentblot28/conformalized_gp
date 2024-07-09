import numpy as np
import openturns as ot
from sklearn.base import BaseEstimator


class GpOTtoSklearnStd(BaseEstimator):
    """
    Wrapper for OpenTURNS Gaussian Process to be used in MAPIE.
    """
    def __init__(
        self, scale: int, amplitude: float,
        nu: float, nugget: bool = False
    ) -> None:
        self.scale = scale
        self.amplitude = amplitude
        self.nu = nu
        self.trained_ = False
        self.nugget = nugget

    def fit(self, X_train, y_train):

        input_dim = X_train.shape[1]
        scale = input_dim * [self.scale]
        amplitude = [self.amplitude]

        covarianceModel = ot.MaternModel(scale, amplitude, self.nu)
        
        if self.nugget:
            covarianceModel.activateNuggetFactor(True)
        basis = ot.ConstantBasisFactory(input_dim).build()

        self.gp = ot.KrigingAlgorithm(
            ot.Sample(X_train),
            ot.Sample(y_train.reshape(-1, 1)),
            covarianceModel, basis
        )
        if self.nugget:
            self.gp.setOptimizationAlgorithm(ot.NLopt("GN_DIRECT"))

        self.gp.run()

        self.trained_ = True

    def predict(self, X_test, return_std=False):

        metamodel = self.gp.getResult()(X_test)

        y_pred = metamodel.getMean()
        y_std = metamodel.getStandardDeviation()

        if not return_std:
            return np.array(y_pred)
        else:
            return np.array(y_pred), np.array(y_std)

    def __sklearn_is_fitted__(self):
        if self.trained_:
            return True
        else:
            return False
