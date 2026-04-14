
import os
os.environ["KERAS_BACKEND"] = "torch"   # must be set before `import keras`

import glob
import time
import numpy as np
from numpy.linalg import cholesky, inv, norm, pinv, lstsq
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau, spearmanr, iqr

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, ExpSineSquared, WhiteKernel, ConstantKernel, Kernel, Hyperparameter
from sklearn.svm import SVR, SVC
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import keras
from keras import layers
import torch
from keras import optimizers, losses, metrics

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
        elif model_name == "MyModel":
            model = MyModel(self.args.problem_id, self.args.load_metadata, self.args.save_metadata)
        elif model_name == "PFNModel":
            model = PFNModel()
        elif model_name == "LocalQuadraticModel":
            model = LocalQuadraticModel()
        elif model_name == "LocalNeuralModel":
            model = LocalNeuralModel()
            
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
                try:
                    pred = self.models[i].predict(data["xs"])
                    err = np.mean((np.array(data['true']) - np.array(pred)) ** 2)
                    self.errors[i] = err
                except Exception as e:
                    self.errors[i] = 0.0
                
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

    def save(self):
        for model in self.models:
            if hasattr(model, "save"):
                model.save()

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

class LocalNeuralModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.X = []
        self.y = []
        self.k = 40
        self.train_X = []
        self.train_y = []

    def build_mlp(self, in_features, out_features=1, hidden=(64, 32), activation="relu", dropout=0.2, batchnorm=True):
        inp = keras.Input(shape=(in_features,))
        x = inp
        for h in hidden:
            x = layers.Dense(h, activation=None,
                             kernel_initializer="he_normal",
                             bias_initializer="zeros")(x)
            if batchnorm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            if dropout and dropout > 0:
                x = layers.Dropout(dropout)(x)
        out = layers.Dense(out_features, activation=None,
                           kernel_initializer="he_normal",
                           bias_initializer="zeros")(x)
        model = keras.Model(inp, out)
        return model

    def save(self):
        np.save(f"data/train_X_{self.X.shape[1]}_{self.k}.npy", np.asarray(self.train_X))
        np.save(f"data/train_y_{self.X.shape[1]}_{self.k}.npy", self.train_y)
        print(f"saving {np.asarray(self.train_X).shape} X")

    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        self.tree = BallTree(self.X, metric="euclidean")

    def predict(self, X):
        X = np.asarray(X)
        # initialize model
        if self.model is None:
            in_features = (X.shape[1] + 1) * self.k
            #self.model = build_transformer(
            #    in_features=in_features, out_features=1,
            #    k=self.k,            # <-- important, sequence length
            #    d_model=128,
            #    depth=4,
            #    num_heads=4,
            #    mlp_ratio=2.0,
            #    dropout=0.1,
            #)
            self.model = self.build_mlp(in_features=in_features, out_features=1, hidden=(1024, 1024), 
                activation="relu", dropout=0.2, batchnorm=True)
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=3e-4),
                loss="mse",
                metrics=["mse"]
            )
            if os.path.exists(f"data/train_X_{X.shape[1]}_{self.k}.npy"):
                self.train_X = np.load(f"data/train_X_{X.shape[1]}_{self.k}.npy")
                self.train_y = np.load(f"data/train_y_{X.shape[1]}_{self.k}.npy")
                train_len = 2 * len(self.train_X) // 3
                X_tr, y_tr = self.train_X[:train_len], self.train_y[:train_len]
                X_va, y_va = self.train_X[train_len:], self.train_y[train_len:]
                self.model.fit(X_tr, y_tr, validation_data=(X_va, y_va), batch_size=128, epochs=5, verbose=1)
                print(f"loaded {self.train_X.shape} X")
                self.train_X = list(self.train_X)
                self.train_y = list(self.train_y)
                
        # preprocess data
        preds = []
        for x in X:
            dist, idx = self.tree.query([x], k=self.k)
            x_coords = self.X[idx][0] - x
            x_std = np.std(x_coords, axis = 0)
            x_std[np.where(x_std < 1e-6)] = 1e-6
            x_coords /= x_std

            y_values = self.y[idx]
            y_mean = np.mean(y_values)
            y_std = max(np.std(y_values), 1e-6)
            y_values = (y_values - y_mean) / y_std
            
            inpt = np.asarray([np.column_stack((x_coords, y_values[0][..., None])).reshape(-1)])

            inpt = torch.asarray(inpt.astype(np.float32, copy=False), device = self.device)
            self.model.eval()
            with torch.no_grad():
                out = self.model(inpt)
            out = out[0,0] * y_std + y_mean
            preds.append(out)
        preds = np.asarray(preds)
        return preds

    def update(self, data):
        if "xs" in data.keys():
            for x in data["xs"]:
                dist, idx = self.tree.query([x], k=self.k)
                x_coords = self.X[idx][0] - x
                x_std = np.std(x_coords, axis = 0)
                x_std[np.where(x_std < 1e-6)] = 1e-6
                x_coords /= x_std

                y_values = self.y[idx]
                y_mean = np.mean(y_values)
                y_std = max(np.std(y_values), 1e-6)
                y_values = (y_values - y_mean) / y_std
                
                inpt = np.asarray([np.column_stack((x_coords, y_values[0][..., None])).reshape(-1)])

                inpt = torch.asarray(inpt.astype(np.float32, copy=False), device = self.device)

                self.train_X.append(np.asarray(inpt[0]))
            for y in data["true"]:
                self.train_y.append(np.asarray((y - y_mean) / y_std))

class LearnablePositionalEncoding(layers.Layer):
    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        # (seq_len, d_model) learned positional table
        self.pos = self.add_weight(
            name="pos_emb", shape=(seq_len, d_model),
            initializer="zeros", trainable=True
        )

    def call(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pos

def transformer_block(x, d_model=128, num_heads=4, mlp_ratio=2.0, dropout=0.1):
    # PreNorm + MHA
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)(y, y)
    y = layers.Dropout(dropout)(y)
    x = layers.Add()([x, y])

    # PreNorm + FFN
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Dense(int(d_model * mlp_ratio), activation="gelu")(y)
    y = layers.Dropout(dropout)(y)
    y = layers.Dense(d_model)(y)
    y = layers.Dropout(dropout)(y)
    x = layers.Add()([x, y])
    return x

def build_transformer(
    in_features: int,         # = k * (d + 1)
    out_features: int = 1,    # regression head by default
    *,
    k: int = 10,              # number of neighbors (sequence length)
    d_model: int = 128,
    depth: int = 2,
    num_heads: int = 4,
    mlp_ratio: float = 2.0,
    dropout: float = 0.1,
    pooling: str = "mean"     # "mean" or "cls"
) -> keras.Model:
    """
    Expects a flat input of shape (k * (d+1),), reshapes to (k, d+1), then runs a Transformer encoder.
    """
    assert in_features % k == 0, f"in_features={in_features} must be divisible by k={k}"
    feat_dim = in_features // k   # = d + 1

    inp = keras.Input(shape=(in_features,), name="flat_seq")
    x = layers.Reshape((k, feat_dim), name="reshape_to_tokens")(inp)     # (B, k, d+1)
    x = layers.Dense(d_model, activation=None, name="token_proj")(x)     # (B, k, d_model)

    # Learnable positional encoding
    x = LearnablePositionalEncoding(seq_len=k, d_model=d_model, name="pos_encoding")(x)

    # Stacked Transformer encoder blocks
    for i in range(depth):
        x = transformer_block(x, d_model=d_model, num_heads=num_heads,
                              mlp_ratio=mlp_ratio, dropout=dropout)

    # Pooling over the sequence
    if pooling == "mean":
        x = layers.GlobalAveragePooling1D(name="gap")(x)   # (B, d_model)
    elif pooling == "cls":
        # prepend a CLS token (optional variant)
        raise NotImplementedError("CLS pooling not wired; use pooling='mean' or extend with a CLS token.")
    else:
        raise ValueError("pooling must be 'mean' or 'cls'")

    # Regression/classification head
    out = layers.Dense(out_features, activation=None, name="head")(x)
    model = keras.Model(inp, out, name="local_transformer")
    return model

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

import torch
from tabpfn import TabPFNClassifier

class PFNModel:
    def __init__(self, max_bins=10):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TabPFNClassifier(device=self.device)
        self.max_bins = max_bins
        self.bin_edges = None
        self.is_fitted = False
        self.X_train = None
        self.y_train = None

    def _bin_data(self, y):
        # bins = min(max_bins, unique values)
        bins = min(self.max_bins, max(2, int(len(y) / 2)))
        # Ensure monotonic bins
        self.bin_edges = np.linspace(np.min(y), np.max(y), bins + 1)
        y_bins = np.digitize(y, self.bin_edges) - 1
        return np.clip(y_bins, 0, bins - 1)

    def fit(self, X, y):
        if len(X) < 3:
            self.is_fitted = False
            return
        y_bins = self._bin_data(y)
        self.model.fit(X, y_bins)  # regular sklearn API
        self.is_fitted = True

    def predict(self, X, return_std=False):
        if not self.is_fitted:
            mean = np.zeros(len(X))
            var = np.full(len(X), 1e3)
            return (mean, var) if return_std else mean

        preds = self.model.predict_proba(X)
        bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        
        # Align class dims between preds and bins
        Kp = preds.shape[1]
        Kb = len(bin_centers)
        K = min(Kp, Kb)

        preds = preds[:, :K]
        centers = bin_centers[:K]

        mean = preds @ centers

        diff = centers - mean[:, None]
        var = np.sum(preds * diff**2, axis=1)

        return (mean, var) if return_std else mean
    
if False:
    rbf_kernel = ConstantKernel() * RBF()
    mat_kernel = ConstantKernel() * Matern(nu=2.5)
    rbf_model = GaussianProcessRegressor(
        kernel=rbf_kernel,
        alpha=1e-4,
        normalize_y=True,
        n_restarts_optimizer=8,
    )
    mat_model = GaussianProcessRegressor(
        kernel=mat_kernel,
        alpha=1e-4,
        normalize_y=True,
        n_restarts_optimizer=8,
    )
    pfn_model = PFNModel()
    rbf_model = RBFModel()
    svr_model = SVR(kernel="rbf", C=100, gamma="scale")
    rf_model = RandomForestRegressor()

class MyModel:
    def __init__(self, problem_id, load_metadata, save_metadata):
        self.problem_id = problem_id
        self.load_metadata = load_metadata
        self.save_metadata = save_metadata
        rbf_kernel = ConstantKernel() * RBF()
        mat_kernel = ConstantKernel() * Matern(nu=2.5)
        rbf_model = GaussianProcessRegressor(
            kernel=rbf_kernel,
            alpha=1e-4,
            normalize_y=True,
            n_restarts_optimizer=8,
        )
        mat_model = GaussianProcessRegressor(
            kernel=mat_kernel,
            alpha=1e-4,
            normalize_y=True,
            n_restarts_optimizer=8,
        )

        self.models = [
            LQModel("linear", change_model=False),
            LQModel("quad", change_model=False),
            RBFModel(),
            SVR(kernel="rbf", C=100, gamma="scale"),
            RandomForestRegressor(),
            #rbf_model,
            #mat_model,
        ]

        self.best_model_index = 0

        # Meta-model: predikuje pravděpodobnost, že model i bude nejlepší
        self.meta_model = LinearRegression()
        #self.meta_model = Ridge(alpha=1e-3)
        #self.meta_model = RandomForestRegressor()
        self.meta_model = Pipeline([
            ("scaler", StandardScaler()),
        #    #("ridge", Ridge(alpha=1e-3))
            ("linreg", LinearRegression())
        ])

        self.meta_X = []
        self.meta_y = []
        if self.load_metadata:
            self.load()

        self.last_ela = None
        self.last_X = []
        self.last_y = []
        self.own_meta_X = []
        self.own_meta_y = []

    def ela(self, X, y):
        n, d = X.shape
        y = np.asarray(y)

        features = []

        # =====================
        # A) Size-related
        # =====================
        features.append(n)
        features.append(d)
        features.append(n / max(d, 1))

        # =====================
        # B) y distribution
        # =====================
        var_y = np.var(y)
        features.append(np.std(y))
        features.append(iqr(y))

        # =====================
        # C) Model-wise quality
        # =====================
        rank_corrs = []
        nmses = []

        for model in self.models:
            try:
                y_pred = model.predict(X)

                # rank correlation
                rho, _ = spearmanr(y, y_pred)
                rho = 0.0 if np.isnan(rho) else rho

                # normalized MSE (scale invariant)
                mse = np.mean((y - y_pred) ** 2)
                nmse = mse / (var_y + 1e-12)

            except Exception:
                rho = 0.0
                nmse = 1.0

            rank_corrs.append(rho)
            nmses.append(nmse)

            features.append(rho)
            features.append(nmse)

        # =====================
        # D) Relative comparison
        # =====================
        rank_corrs = np.asarray(rank_corrs)
        best = np.max(rank_corrs)
        second = np.partition(rank_corrs, -2)[-2] if len(rank_corrs) > 1 else 0.0

        features.append(best)
        features.append(second)
        features.append(best - second)

        # =====================
        # E) Stability (best model)
        # =====================
        best_idx = int(np.argmax(rank_corrs))
        try:
            y_pred_best = self.models[best_idx].predict(X)
            tau, _ = kendalltau(y, y_pred_best)
            tau = 0.0 if np.isnan(tau) else tau
        except Exception:
            tau = 0.0

        features.append(tau)

        return np.asarray(features, dtype=float)
    
    def update(self, data):
        if "xs" not in data.keys() or len(self.last_X) == 0:
            return
        
        self.meta_X.append(self.last_ela)
        self.own_meta_X.append(self.last_ela)        

        X = np.asarray(data["xs"])
        true = np.asarray(data["true"])

        losses = []
        for model in self.models:
            start_time = time.time()
            model.fit(self.last_X, self.last_y)
            print(f"{model}: {time.time() - start_time:.2f}s (fit)")
            y_pred = model.predict(X)
            print(f"{model}: {time.time() - start_time:.2f}s (predict)")
            rho, _ = spearmanr(true, y_pred)
            if np.isnan(rho):
                rho = 0.0            
            losses.append(1.0 - rho)

        self.meta_y.append(losses)
        self.own_meta_y.append(losses)

        # uč se až když máš dost dat
        if len(self.meta_X) >= 10:
            self.meta_model.fit(
                np.asarray(self.meta_X),
                np.asarray(self.meta_y)
            )
        #print(f"update: {self.best_model_index}")

    def fit(self, X, y, weights=None):
        X_np = np.asarray(X)
        y_np = np.asarray(y)
        self.last_X = X_np
        self.last_y = y_np
        self.last_ela = self.ela(X_np, y_np)

        if len(self.meta_X) >= 10:
            predicted_losses = self.meta_model.predict(
                self.last_ela.reshape(1, -1)
            )[0]
            print(predicted_losses)
            self.best_model_index = int(np.argmin(predicted_losses))
        else:
            # cold start
            self.best_model_index = 0

        #print(f"fit: {self.best_model_index}")
        self.models[self.best_model_index].fit(X_np, y_np)

    def predict(self, X):
        X_np = np.asarray(X)
        #print(f"predict: {self.best_model_index}")
        pred = self.models[self.best_model_index].predict(X_np)
        #print(pred)
        return pred
    
    def load(self):
        self.meta_X = []
        self.meta_y = []

        if not os.path.isdir("data"):
            return False

        pattern = os.path.join("data", "meta-data-*.npz")
        files = glob.glob(pattern)
        func_id = self.problem_id.split("_")[1]

        for fname in files:
            # očekáváme tvar: meta-data-{problem_id}.npz
            try:
                pid = fname.split("meta-data-")[1].split(".npz")[0]
                fid = pid.split("_")[1]
            except Exception:
                continue

            # přeskoč aktuální problém (test leakage!)
            if str(fid) == str(func_id):
                continue

            data = np.load(fname)

            meta_X = data["meta_X"]
            meta_y = data["meta_y"]

            # append po řádcích
            for x, y in zip(meta_X, meta_y):
                self.meta_X.append(x)
                self.meta_y.append(y)

        #self.meta_X, self.meta_y = self.stratified_subsample(self.meta_X, self.meta_y, n_per_bin=500, n_bins=10)
        #self.meta_X, self.meta_y = list(self.meta_X), list(self.meta_y)

        if len(self.meta_X) >= 10:
            self.meta_model.fit(
                np.asarray(self.meta_X),
                np.asarray(self.meta_y)
            )

        return len(self.meta_X) > 0
    
    def save(self):
        if self.save_metadata:
            os.makedirs("data", exist_ok=True)

            filename = f"data/meta-data-{self.problem_id}.npz"

            np.savez_compressed(
                filename,
                meta_X=np.asarray(self.own_meta_X),
                meta_y=np.asarray(self.own_meta_y),
            )

class LQModel:
    def __init__(self, model_type = "linear", change_model = True):
        self.coeffs = None
        self.model_type = model_type
        self.change_model = change_model

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
        if self.change_model:
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
