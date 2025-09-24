
import numpy as np
from numpy.linalg import cholesky, inv, norm, pinv, lstsq
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, ExpSineSquared, WhiteKernel, ConstantKernel, Kernel, Hyperparameter
from sklearn.svm import SVR, SVC

class Model:
    def __init__(self, args):
        self.args = args
        self.models = []
        self.errors = []
        for model_name in args.models.split(","):
            model = self.get_model(model_name)
            if model is not None:
                self.models.append(model)
                self.errors.append(1.0)
        if self.args.ensemble_type == "actor-critic":
            self.alpha = 0.1
            self.beta = 0.1
            self.gamma = 0.95
            self.policy = np.ones(len(self.models)) / len(self.models)
            self.value = 0.0
            self.last_action = 0

    def is_empty(self):
        return len(self.models) == 0

    def get_model(self, model_name):
        model = None

        if "GaussianProcessRegressor" in model_name:
            model = self.get_gpr(model_name)
        elif model_name == "SVC":
            model = SVC(kernel = "linear")
        elif model_name == "RBFModel":
            model = RBFModel()
        elif model_name == "SVR":
            model = SVR(kernel="rbf", C=100, gamma="scale")
        elif model_name == "QuadraticRegression":
            model = QuadraticRegression()
        elif model_name == "LQModel":
            model = LQModel()
        elif model_name == "LocalQuadraticModel":
            model = LocalQuadraticModel()
            
        return model

    def get_gpr(self, model_name):
        kernel_name = model_name.replace("GaussianProcessRegressor", "")
        
        if kernel_name == "LIN":
            kernel = DotProduct()
        elif kernel_name == "Q":
            kernel = DotProduct() ** 2
        elif kernel_name == "SE":
            kernel = ConstantKernel() * RBF()
        elif kernel_name == "MAT":
            kernel = ConstantKernel() * Matern(nu=2.5)
        elif kernel_name == "RQ":
            kernel = ConstantKernel() * RationalQuadratic()
        elif kernel_name == "NN":
            kernel = ConstantKernel() * NeuralNetworkKernel()
        elif kernel_name == "GIBBS":
            kernel = ConstantKernel() * GibbsKernel()
        elif kernel_name == "SEQ":
            kernel = ConstantKernel() * RBF() + DotProduct() ** 2
        elif kernel_name == "MY":
            kernel = ConstantKernel(1.0, (1e-3, 1e8)) * (
                Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e4), nu=2.5)
                + RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e4))
            )
        elif kernel_name == "SAACM":
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(self.args.dimension), length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-1))
            
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-4,
            normalize_y=True,
            n_restarts_optimizer=8,
        )

    def update(self, data):
        if "xs" in data.keys():
            if len(data["xs"]) == 0:
                return
            for i in range(len(self.models)):
                pred = self.models[i].predict(data["xs"])
                err = np.mean((np.array(data['true']) - np.array(pred)) ** 2)
                self.errors[i] = err
                
            if self.args.ensemble_type == "actor-critic":
                reward = -np.min(data['true'])
                # Temporal Difference error
                td_error = reward + self.gamma * self.value - self.value
                self.value += self.beta * td_error
                # Policy update
                one_hot = np.zeros(len(self.models))
                one_hot[self.last_action] = 1.0
                self.policy += self.alpha * td_error * (one_hot - self.policy)
                self.policy = np.clip(self.policy, 1e-6, 1.0)  # keep probs valid
                self.policy /= np.sum(self.policy)
                
        if "es" in data.keys():
            for i in range(len(self.models)):
                if hasattr(self.models[i], "update"):
                    self.models[i].update(data)

    def fit(self, X, y):
        """Fit cubic RBF: phi(r) = r^3"""
        for model in self.models:
            model.fit(X, y)

    def predict(self, X, return_std = False):
        """Predict using cubic RBF"""
        if self.args.ensemble_type == "mean":
            ys = []
            for model in self.models:
                if return_std:
                    ys.append(model.predict(X, return_std = True))
                else:
                    ys.append(model.predict(X))
            return np.mean(np.asarray(ys), axis = 0)
        elif self.args.ensemble_type == "best":
            if return_std:
                return self.models[np.argmin(self.errors)].predict(X, return_std = True)
            else:
                return self.models[np.argmin(self.errors)].predict(X)
        elif self.args.ensemble_type == "weighted":
            eps = 1e-8
            inv_errors = 1.0 / (np.asarray(self.errors) + eps)
            weights = inv_errors / np.sum(inv_errors)

            preds = []
            if return_std:
                # každé predict vrací tuple (mean, std)
                means, stds = [], []
                for model in self.models:
                    mean, std = model.predict(X, return_std=True)
                    means.append(mean)
                    stds.append(std)
                means = np.asarray(means)
                stds = np.asarray(stds)

                weighted_mean = np.tensordot(weights, means, axes=1)
                # vážený průměr rozptylů (approx.)
                weighted_var = np.tensordot(weights, stds**2 + means**2, axes=1) - weighted_mean**2
                weighted_std = np.sqrt(np.maximum(weighted_var, 1e-12))
                return weighted_mean, weighted_std
            else:
                for model in self.models:
                    preds.append(model.predict(X))
                preds = np.asarray(preds)
                return np.tensordot(weights, preds, axes=1)
        elif self.args.ensemble_type == "actor-critic":
            model_i = np.random.choice(len(self.models), p=self.policy)
            self.last_action = model_i
            if return_std:
                return self.models[model_i].predict(X, return_std = True)
            else:
                return self.models[model_i].predict(X)

    def decision_function(self, X):
        """Predict using cubic RBF"""
        if self.args.ensemble_type == "mean":
            ys = []
            for model in self.models:
                ys.append(model.decision_function(X))
            return np.mean(np.asarray(ys), axis = 0)
        elif self.args.ensemble_type == "best":
            return self.models[np.argmin(self.errors)].decision_function(X)
        elif self.args.ensemble_type == "weighted":
            eps = 1e-8
            inv_errors = 1.0 / (np.asarray(self.errors) + eps)
            weights = inv_errors / np.sum(inv_errors)

            preds = []
            for model in self.models:
                preds.append(model.decision_function(X))
            preds = np.asarray(preds)
            return np.tensordot(weights, preds, axes=1)
        elif self.args.ensemble_type == "actor-critic":
            model_i = np.random.choice(len(self.models), p=self.policy)
            self.last_action = model_i
            return self.models[model_i].decision_function(X)

class LinearQuadraticRegression:
    def __init__(self):
        self.mean_y = None
        self.std_y = None
        self.coef_ = None
        self.dim = None

    def _expand_features(self, X):
        """
        Build feature matrix:
        [1, x1, ..., xd, x1^2, ..., xd^2, x1*x2, ...]
        """
        n, d = X.shape
        features = [np.ones((n, 1)), X]  # bias + linear

        # squared terms
        features.append(X ** 2)

        # cross terms
        cross_terms = []
        for i in range(d):
            for j in range(i + 1, d):
                cross_terms.append((X[:, i] * X[:, j])[:, None])
        if cross_terms:
            features.append(np.hstack(cross_terms))

        return np.hstack(features)

    def fit(self, X, y):
        """
        Fit linear+quadratic regression model.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.dim = X.shape[1]

        Phi = self._expand_features(X)

        # Least squares solution
        self.coef_, *_ = np.linalg.lstsq(Phi, y, rcond=None)

    def predict(self, X):
        """
        Predict values for new samples.
        """
        if self.coef_ is None:
            raise RuntimeError("LinearQuadraticRegression not fitted yet.")

        X = np.asarray(X)

        Phi = self._expand_features(X)
        y_pred = Phi @ self.coef_

        return y_pred


class QuadraticRegression:
    def __init__(self):
        self.coef_ = None  # regression coefficients
        self.dim = None    # input dimension

    def _expand(self, X):
        """
        Expand input features with quadratic terms:
        [1, x1, x2, ..., xd, x1^2, x1*x2, ..., xd^2]
        """
        n, d = X.shape
        features = [np.ones((n, 1)), X]  # bias + linear terms

        # quadratic terms
        quad_terms = []
        for i in range(d):
            for j in range(i, d):
                quad_terms.append((X[:, i] * X[:, j])[:, None])
        if quad_terms:
            features.append(np.hstack(quad_terms))

        return np.hstack(features)

    def fit(self, X, y):
        """
        Fit quadratic regression model.
        X: (n_samples, n_features)
        y: (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.dim = X.shape[1]

        X_exp = self._expand(X)
        # Solve least squares: beta = (X^T X)^(-1) X^T y
        self.coef_, *_ = np.linalg.lstsq(X_exp, y, rcond=None)

    def predict(self, X):
        """
        Predict using the fitted quadratic model.
        X: (n_samples, n_features)
        """
        if self.coef_ is None:
            raise RuntimeError("QuadraticRegression not fitted yet.")
        X = np.asarray(X)
        X_exp = self._expand(X)
        return X_exp @ self.coef_

class LocalQuadraticModel:
    """
    Local Quadratic Model with Locally Weighted Regression (LWR).
    Compatible with sklearn interface (init, fit, predict).
    """

    def __init__(self, k_neighbors=40, C=None):
        self.k_neighbors = k_neighbors
        self.C = C

    def _features(self, X):
        """Construct quadratic feature expansion with cross terms."""
        X = np.atleast_2d(X)
        n, d = X.shape
        feats = [np.ones((n, 1)), X, X**2]
        cross = []
        for i in range(d):
            for j in range(i + 1, d):
                cross.append((X[:, i] * X[:, j])[:, None])
        if cross:
            feats.append(np.hstack(cross))
        return np.hstack(feats)

    def fit(self, X, y):
        """
        Store training data for local regression.
        """
        self.X_ = np.array(X)
        self.y_ = np.array(y)
        if self.C is None:
            self.C = np.eye(self.X_.shape[1])
        return self

    def predict(self, Q):
        if not hasattr(self, "X_"):
            raise RuntimeError("Model must be fitted before calling predict().")

        X, y = self.X_, self.y_
        Q = np.atleast_2d(Q)

        # Mahalanobis distances
        dists = cdist(Q, X, metric="mahalanobis", VI=inv(self.C))

        preds = []
        for qi, di in zip(Q, dists):
            # nearest k neighbors
            idx = np.argsort(di)[: self.k_neighbors]
            Xi, yi, di = X[idx], y[idx], di[idx]
            h = di[-1] + 1e-12
            w = (1 - (di / h) ** 2) ** 2
            W = np.sqrt(w)
            Z = self._features(Xi) * W[:, None]
            yw = yi * W
            beta, *_ = lstsq(Z, yw, rcond=None)
            preds.append(self._features(qi) @ beta)

        return np.array(preds)

    def update(self, data):
        self.C = data['es'].sm.C

class LQModel:
    def __init__(self):
        self.coeffs = None
        self.model_type = "linear"

    def _features(self, X):
        X = np.atleast_2d(X)
        feats = [np.ones((X.shape[0], 1)), X]
        if self.model_type in ["quad", "full"]:
            feats.append(X**2)
        if self.model_type == "full":
            cross = [np.prod(X[:, [i,j]], axis=1, keepdims=True)
                     for i in range(self.dim) for j in range(i+1, self.dim)]
            feats.append(np.hstack(cross))
        return np.hstack(feats)

    def fit(self, X, y, weights=None):
        self.dim = X.shape[1]
        self.model_type = "linear" if len(X) < 2*self.dim \
            else "quad" if len(X) < (self.dim**2) \
            else "full"
        Z = self._features(X)
        if weights is None:
            weights = np.ones(len(y))
        W = np.diag(weights)
        self.coeffs = pinv(W @ Z) @ (W @ y)

    def predict(self, X):
        return self._features(X) @ self.coeffs

class RBFModel:
    def __init__(self):
        self.coeff = None
        self.centers = None
        self.poly = None

    def fit(self, X, y):
        """Train cubic RBF surrogate (with polynomial tail)."""
        n, d = X.shape
        self.centers = X
        Phi = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                r = norm(X[i] - X[j])
                Phi[i, j] = r**3

        # Polynomial tail [1, x]
        P = np.hstack([np.ones((n, 1)), X])

        # Build system
        A = np.block([[Phi, P],
                      [P.T, np.zeros((d+1, d+1))]])
        b = np.concatenate([y, np.zeros(d+1)])

        sol = np.linalg.solve(A, b)
        self.coeff, self.poly = sol[:n], sol[n:]

    def predict(self, Xq):
        """Evaluate surrogate at query points Xq (N,d)."""
        n, d = self.centers.shape
        ypred = []
        for x in Xq:
            r = norm(self.centers - x, axis=1)
            Phi = r**3
            ypred.append(np.dot(self.coeff, Phi) + self.poly[0] + np.dot(self.poly[1:], x))
        return np.array(ypred)


class NeuralNetworkKernel(Kernel):
    """Neural Network (arcsine) kernel from Rasmussen & Williams (2006)."""

    def __init__(self, sigma_f=1.0):
        self.sigma_f = sigma_f
        # define as hyperparameter so sklearn can tune it
        self.sigma_f_bounds = (1e-5, 1e5)

    @property
    def hyperparameters(self):
        return [Hyperparameter("sigma_f", "numeric", self.sigma_f_bounds)]

    @property
    def theta(self):
        return np.log([self.sigma_f])

    @theta.setter
    def theta(self, theta):
        self.sigma_f = np.exp(theta[0])

    @property
    def bounds(self):
        return np.log([self.sigma_f_bounds])

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        norm_X = np.sqrt(1 + np.sum(X**2, axis=1))[:, None]
        norm_Y = np.sqrt(1 + np.sum(Y**2, axis=1))[None, :]
        prod = X @ Y.T
        K = (self.sigma_f**2) * np.arcsin(prod / (norm_X * norm_Y))

        if eval_gradient:
            # gradient wrt sigma_f only
            grad = (2 * self.sigma_f) * np.arcsin(prod / (norm_X * norm_Y))
            return K, grad[:, :, None]
        return K

    def diag(self, X):
        return np.full(X.shape[0], self.sigma_f**2 * np.arcsin(1.0))

    def is_stationary(self):
        return False

class GibbsKernel(Kernel):
    """Non-stationary Gibbs kernel with finite bounds."""

    def __init__(self, sigma_f=1.0, ell0=1.0, gamma=0.0,
                 sigma_f_bounds=(1e-3, 1e3),
                 ell0_bounds=(1e-3, 1e3),
                 gamma_bounds=(-5.0, 5.0)):
        self.sigma_f = sigma_f
        self.ell0 = ell0
        self.gamma = gamma
        self.sigma_f_bounds = sigma_f_bounds
        self.ell0_bounds = ell0_bounds
        self.gamma_bounds = gamma_bounds

    # --- sklearn API ---
    @property
    def hyperparameter_sigma_f(self):
        return Hyperparameter("sigma_f", "numeric", self.sigma_f_bounds)

    @property
    def hyperparameter_ell0(self):
        return Hyperparameter("ell0", "numeric", self.ell0_bounds)

    @property
    def hyperparameter_gamma(self):
        return Hyperparameter("gamma", "numeric", self.gamma_bounds)

    @property
    def theta(self):
        # log-transform hyperparameters
        return np.log([self.sigma_f, self.ell0, self.gamma + 1e-6])  # avoid log(0)

    @theta.setter
    def theta(self, theta):
        # inverse log-transform
        self.sigma_f, self.ell0, gamma = np.exp(theta)
        self.gamma = gamma - 1e-6  # restore gamma shift

    @property
    def bounds(self):
        return np.log([
            self.sigma_f_bounds,
            self.ell0_bounds,
            (max(1e-6, self.gamma_bounds[0] + 1e-6), self.gamma_bounds[1] + 1e-6)
        ])

    # --- Kernel logic ---
    def _ell(self, X):
        r2 = np.sum(X**2, axis=1)
        return self.ell0 * np.exp(0.5 * self.gamma * r2)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        D = X.shape[1]

        ell_X = self._ell(X)
        ell_Y = self._ell(Y)

        d2 = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2.0 * X @ Y.T
        S = (ell_X**2)[:, None] + (ell_Y**2)[None, :]

        P = (2.0 * (ell_X[:, None] * ell_Y[None, :]) / np.maximum(S, 1e-12)) ** (0.5 * D)
        E = np.exp(-d2 / np.maximum(S, 1e-12))

        K = (self.sigma_f ** 2) * P * E

        if eval_gradient:
            grad_sigma_f = 2 * K / self.sigma_f
            grad_ell0 = np.zeros_like(K)
            grad_gamma = np.zeros_like(K)
            return K, np.stack([grad_sigma_f, grad_ell0, grad_gamma], axis=2)

        return K

    def diag(self, X):
        return np.full(X.shape[0], self.sigma_f ** 2)

    def is_stationary(self):
        return False

    def __repr__(self):
        return f"GibbsKernel(sigma_f={self.sigma_f:.3g}, ell0={self.ell0:.3g}, gamma={self.gamma:.3g})"
