
from sklearn.preprocessing import StandardScaler as StdScaler
from sklearn.preprocessing import MinMaxScaler as MiMaScaler
from sklearn.preprocessing import PolynomialFeatures as PolyScaler
import numpy as np
from numpy.linalg import cholesky, inv, norm

class Scalers:
    @staticmethod
    def get(scaler_name, args):
        scaler = None
        
        if scaler_name == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_name == "DTSScaler":
            scaler = DTSScaler()
        elif scaler_name == "MinMaxScaler":
            scaler = MinMaxScaler()

        return scaler


class StandardScaler:
    def __init__(self):
        self.scaler = StdScaler()

    def fit_transform(self, data):
        if len(data['X'].shape) == 1:
            return self.scaler.fit_transform(data['X'][:,np.newaxis])[:,0]
        return self.scaler.fit_transform(data['X'])

    def transform(self, X):
        if len(X.shape) == 1:
            return self.scaler.transform(X[:,np.newaxis])[:,0]
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        if len(X.shape) == 1:
            return self.scaler.inverse_transform(X[:,np.newaxis])[:,0]
        return self.scaler.inverse_transform(X)


class MinMaxScaler:
    def __init__(self):
        self.scaler = MiMaScaler()

    def fit_transform(self, data):
        if len(data['X'].shape) == 1:
            return self.scaler.fit_transform(data['X'][:,np.newaxis])[:,0]
        return self.scaler.fit_transform(data['X'])

    def transform(self, X):
        if len(X.shape) == 1:
            return self.scaler.transform(X[:,np.newaxis])[:,0]
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        if len(X.shape) == 1:
            return self.scaler.inverse_transform(X[:,np.newaxis])[:,0]
        return self.scaler.inverse_transform(X)

class DTSScaler:
    def __init__(self):
        self.cov_inv_sqrt = None
        self.sigma = None
        self.mean = None

    def fit_transform(self, data):
        self.es = data['es']
        self.C = self.es.sm.C
        self.sigma = self.es.sigma
        self.mean = self.es.mean

        try:
            # eigendecomposition of C
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.clip(eigvals, 1e-12, None)  # safeguard
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        except np.linalg.LinAlgError:
            C_inv_sqrt = np.eye(self.C.shape[0])

        # include scaling by 1/sigma^2
        self.cov_inv_sqrt = (1.0 / (self.sigma**2)) * C_inv_sqrt

        return (data['X']) @ self.cov_inv_sqrt.T

    def transform(self, X):
        return (X) @ self.cov_inv_sqrt.T

    def inverse_transform(self, X):
        # Reconstruct: multiply by inverse of (1/sigma^2 * C^-1/2)
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.clip(eigvals, 1e-12, None)
        C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        cov_sqrt = (self.sigma**2) * C_sqrt
        return X @ cov_sqrt.T

class SAOScaler:
    def __init__(self):
        self.C = None
        self.Cinvsqrt = None
        self.mean = None

    def fit_transform(self, data):
        self.C = data['es'].sm.C  # covariance matrix from cma
        self.Cinvsqrt = inv(cholesky(C))
        self.mean = data['es'].mean
        return (data['X'] - self.mean) @ self.Cinvsqrt.T

    def transform(self, X):
        return (X - self.mean) @ self.Cinvsqrt.T

    def inverse_transform(self, X):
        return X @ np.linalg.inv(self.Cinvsqrt.T) + self.mean
