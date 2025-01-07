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



class GpOTtoSklearnChooseKernel(BaseEstimator):
    """
    Wrapper for OpenTURNS Gaussian Process to be used in MAPIE.
    """
    def __init__(
        self, trend: str, kernel: str, 
        dimension:int, noise: float = None
    ) -> None:
        self.trend = trend
        self.kernel = kernel
        self.dimension = dimension
        self.trained_ = False
        self.noise = noise

    def fit(self, X_train, y_train):

        if self.trend not in ['Constant', 'Linear', 'Quad']:
            raise ValueError(f"trend must be one of ['Constant', 'Linear', 'Quad']")

        if self.kernel not in ['AbsExp', 'SqExp', 'M-1/2', 'M-3/2', 'M-5/2']:
            raise ValueError(f"kernel must be one of ['AbsExp', 'SqExp', 'M-1/2', 'M-3/2', 'M-5/2']")
        
        if self.trend == 'Constant':
            basis = ot.ConstantBasisFactory(self.dimension).build()
        elif self.trend == 'Linear':
            basis = ot.LinearBasisFactory(self.dimension).build()
        elif self.trend == 'Quad':
            basis = ot.QuadraticBasisFactory(self.dimension).build()    

        if self.kernel == 'AbsExp':
            covarianceModel = ot.AbsoluteExponential([1.0]*self.dimension)
        elif self.kernel == 'SqExp':
            covarianceModel = ot.SquaredExponential([1.0]*self.dimension)
        elif self.kernel == 'M-1/2':
            covarianceModel = ot.MaternModel([1.0]*self.dimension, [1.0], 0.5)
        elif self.kernel == 'M-3/2':
            covarianceModel = ot.MaternModel([1.0]*self.dimension, [1.0], 1.5)
        elif self.kernel == 'M-5/2':
            covarianceModel = ot.MaternModel([1.0]*self.dimension, [1.0], 2.5)

        if self.noise:
            covarianceModel.setNuggetFactor(self.noise)

        self.gp = ot.KrigingAlgorithm(
            ot.Sample(X_train),
            ot.Sample(y_train.reshape(-1, 1)),
            covarianceModel, basis
        )

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
