import os
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, ExpSineSquared, WhiteKernel, ConstantKernel, Kernel, Hyperparameter
from cma import CMAEvolutionStrategy

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import kendalltau
from scipy.special import softmax
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from model import LQModel

class Features:
    def __init__(self):
        pass

    @staticmethod
    def _cv_mse(est, X, y, n_splits=5):
        n = len(y)
        if n < n_splits:
            return np.mean((y - est.fit(X, y).predict(X))**2)
        scores = -cross_val_score(est, X, y,
                                  scoring="neg_mean_squared_error",
                                  cv=min(n_splits, n))
        return float(np.mean(scores))

    @staticmethod
    def _diag_quadratic_mse(X, y):
        poly = PolynomialFeatures(degree=2, include_bias=True)
        Z = poly.fit_transform(X)
        powers = poly.powers_  # monomials matrix

        keep = []
        for exps in powers:
            nz = np.count_nonzero(exps)
            if nz == 0:
                keep.append(True)  # bias
            elif nz == 1 and (1 <= exps.max() <= 2):
                keep.append(True)  # x_i or x_i^2
            else:
                keep.append(False)
        Zd = Z[:, keep]

        lr = LinearRegression()
        return Features._cv_mse(lr, Zd, y)

    @staticmethod
    def get(X, y):
        """
        Compute landscape features.
        X: (n, d)
        y: (n,)
        Returns: dict of numeric features
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        n, d = X.shape

        feats = {
            "n_samples": float(n),
            "dim": float(d),
        }

        # LR baseline
        lr = LinearRegression()
        mse_lr = Features._cv_mse(lr, X, y)
        lr.fit(X, y)
        yhat_lr = lr.predict(X)
        r2_lr = r2_score(y, yhat_lr)

        feats["mse_lr"] = float(mse_lr)
        feats["r2_lr"] = float(r2_lr)

        # Linearity (directional correlation)
        if np.linalg.norm(lr.coef_) > 0:
            z = X @ lr.coef_
            tau_lin, _ = kendalltau(z, y)
            feats["tau_lin"] = float(0.0 if np.isnan(tau_lin) else tau_lin)
        else:
            feats["tau_lin"] = 0.0

        # Heteroscedasticity proxy
        res = y - yhat_lr
        try:
            bp = LinearRegression().fit(X, res**2)
            r2_bp = r2_score(res**2, bp.predict(X))
            feats["bp_r2"] = float(np.clip(r2_bp, 0.0, 1.0))
        except Exception:
            feats["bp_r2"] = 0.0

        # Curvature
        mse_q = Features._diag_quadratic_mse(X, y)
        feats["mse_qdiag"] = float(mse_q)
        feats["curv_gain"] = float((mse_lr - mse_q) / (abs(mse_lr) + 1e-12))

        return feats

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


class Visualizer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.initialized_pca = False
        self.pca = PCA(n_components=2)

    def update(self, X, y, age):
        self.ax.clear()

        # Fit PCA only once for stability
        if not self.initialized_pca:
            self.pca.fit(X)
            self.initialized_pca = True

        X_2d = self.pca.transform(X)

        # 3D souřadnice
        xs = X_2d[:, 0]
        ys = X_2d[:, 1]
        zs = np.log(np.asarray(y) + 1000)  # hodnoty jako výška

        sc = self.ax.scatter(xs, ys, zs, c=age, cmap="coolwarm")

        self.ax.set_title("3D PCA vizualizace s věkem")
        self.ax.set_xlabel("PC1")
        self.ax.set_ylabel("PC2")
        self.ax.set_zlabel("Fitness (y)")

        #cbar = plt.colorbar(sc, ax=self.ax, pad=0.2)
        #cbar.set_label("Age")

        plt.pause(0.01)


class TransferModel:
    def __init__(self):
        self.alpha = 0.9
        self.train_len = 1000000
        #self.model_names = ["gpr_rbf", "gpr_matern"]
        self.model_names = ["linear", "quad", "full"]
        self.models = [
            LQModel(model_type = "linear", change_model = False),
            LQModel(model_type = "quad", change_model = False),
            LQModel(model_type = "full", change_model = False),
            #GaussianProcessRegressor(
            #    kernel=ConstantKernel() * RBF(),
            #    alpha=1e-4,
            #    normalize_y=True,
            #    n_restarts_optimizer=8,
            #),
            #GaussianProcessRegressor(
            #    kernel=ConstantKernel() * Matern(nu=2.5),
            #    alpha=1e-4,
            #    normalize_y=True,
            #    n_restarts_optimizer=8,
            #)
        ]
        # One meta-model per surrogate model
        self.meta_models = [
            DecisionTreeRegressor(),
            DecisionTreeRegressor(),
            DecisionTreeRegressor(),
        ]
        self.models_dist = np.ones(len(self.models)) / len(self.models)
        self.best_idx = None

        self.L_keys = None
        self.fitted_models = [False] * len(self.models)
        self.fitted_meta = [False] * len(self.models)

        self.best_indices = []
        self.best_fs = []
        self.all_ls = []
        self.Ls = []
        self.accs = []
        for i in range(len(self.models)):
            self.Ls.append([])
            self.accs.append([])

        self.training = False


    def _extract_L(self, L_dict):
        if self.L_keys is None:
            self.L_keys = sorted(L_dict.keys())
        return np.array([L_dict[k] for k in self.L_keys], dtype=float).reshape(1, -1)

    def levenshtein_distance(self, a, b):
        m, n = len(a), len(b)
        dp = [[0]*(n+1) for _ in range(m+1)]
        
        for i in range(m+1): dp[i][0] = i
        for j in range(n+1): dp[0][j] = j
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(dp[i-1][j] + 1,
                               dp[i][j-1] + 1,
                               dp[i-1][j-1] + cost)
        return dp[m][n]

    def kendall_tau_distance(self, rank1, rank2):
        n = len(rank1)
        inv = 0
        pos = {item: idx for idx, item in enumerate(rank2)}
        for i in range(n):
            for j in range(i+1, n):
                if (pos[rank1[i]] > pos[rank1[j]]):
                    inv += 1
        return inv

    def normalized_kendall_distance(self, r1, r2):
        n = len(r1)
        dist = self.kendall_tau_distance(r1, r2)
        max_dist = n*(n-1)//2  # maximum number of pairwise disagreements
        return dist / max_dist

    def save(self):
        SAVE_DIR = "surrogate_data"
        os.makedirs(SAVE_DIR, exist_ok=True)

        for i in range(len(self.models)):
            model_name = self.model_names[i]
            filename_Ls = os.path.join(SAVE_DIR, f"{model_name}-data-Ls.npy")
            filename_accs = os.path.join(SAVE_DIR, f"{model_name}-data-accs.npy")
            np.save(filename_Ls, np.asarray(self.Ls[i]))
            np.save(filename_accs, np.asarray(self.accs[i]))
            print(f"Data of Model {i} saved.")

    def load(self):
        SAVE_DIR = "surrogate_data"
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        for fname in os.listdir(SAVE_DIR):
            if fname.endswith("-data-Ls.npy"):
                model_name = fname.split("-data-Ls.npy")[0]
                self.Ls[self.model_names.index(model_name)] = np.load(os.path.join(SAVE_DIR, fname), allow_pickle=True).tolist()
                print(f"📌 Loaded {fname} as {model_name}")
            if fname.endswith("-data-accs.npy"):
                model_name = fname.split("-data-accs.npy")[0]
                self.accs[self.model_names.index(model_name)] = np.load(os.path.join(SAVE_DIR, fname), allow_pickle=True).tolist()
                print(f"📌 Loaded {fname} as {model_name}")

    # --------------------
    # Update distribution and fit best model
    # --------------------
    def fit(self, X, y, L_dict):
        L = self._extract_L(L_dict)

        # Predict loss for each model using each meta-model
        preds = []
        for i, mm in enumerate(self.meta_models):
            if self.fitted_meta[i]:
                acc_pred = mm.predict(L)[0]
                preds.append(acc_pred)
            else:
                preds.append(0.5)  # neutral prior guess

        preds = np.array(preds)
        self.models_dist = softmax(preds)
        #self.models_dist = self.alpha * self.models_dist + (1-self.alpha) * softmax(preds)
    

        # Choose best model by distribution
        for i in range(len(self.models)):
            self.models[i].fit(X, y)
            self.fitted_models[i] = True

        print(f"✅ FIT: Selected models with prob {self.models_dist}")

    # --------------------
    # Prediction using best current model
    # --------------------
    def predict(self, X):
        if self.training:
            self.best_idx = int(np.random.choice(len(self.models)))
        else:
            self.best_idx = np.argmax(self.models_dist)
        mdl = self.models[self.best_idx]
        if not self.fitted_models[self.best_idx]:
            return np.full(len(X), 1e6)
        return mdl.predict(X)

    # --------------------
    # Update meta-model using observed new losses
    # --------------------
    def update(self, X_obs, y_obs, best_f, L_dict):
        if len(X_obs) < 10:
            return

        L = self._extract_L(L_dict)[0]
        
        self.all_ls.append(L)
        self.best_fs.append(best_f)
        self.best_indices.append(self.best_idx)

        if len(self.best_indices) >= 10:
            i = self.best_indices[-10]

            if self.training:
                self.Ls[i].append(self.all_ls[-10])
                self.accs[i].append(self.best_fs[-10] - self.best_fs[-1])
            
            self.meta_models[i].fit(self.Ls[i], self.accs[i])
            self.fitted_meta[i] = True

        #print(f"📌 UPDATE: Model {self.best_idx} better up = {better_up:.4f}")

class ReinforcementOptimizer:
    """
    Surrogate-assisted CMA-ES:
    1) CMA asks for λ proposals
    2) GP surrogate predicts fitness for all proposals
    3) Take top fraction according to args.real_evaluation_ratio
    4) Evaluate those selected proposals on the real problem
    5) Update CMA-ES only using real evaluations
    """

    def __init__(self):
        self.x_obs = None
        self.y_obs = None
        self.ages = None

    def optimize(self, problem, args):
        dim = args.dimension
        max_evals = args.max_evals_per_dim * dim
        popsize = args.pop_size
        sigma0 = args.sigma

        # CMA-ES initialization
        x0 = np.zeros(dim)  # default as in args.init = "zeros"
        es = CMAEvolutionStrategy(x0, sigma0,
                                  {"popsize": popsize})

        # GP surrogate model
        features = Features()
        model = TransferModel()
        model.load()

        best_so_far = None
        n_real_evals = 0
        it = 0

        #vis = Visualizer()

        while not es.stop() and n_real_evals < max_evals:

            X = np.array(es.ask())
            ratio = args.real_evaluation_ratio.get()

            L_dict = None
            if self.x_obs is not None:
                L_dict = features.get(self.x_obs, self.y_obs)
                model.fit(self.x_obs, self.y_obs, L_dict)
                
            if self.x_obs is not None and ratio < 1.0 and kendalltau(self.y_obs, model.predict(self.x_obs))[0] > 0.85:
                y_pred = model.predict(X)
                
                k = max(1, int(popsize * ratio))
                best_ids = np.argsort(y_pred)[:k]
                X_real = X[best_ids]
                y_pred_chosen = y_pred[best_ids]
                y = [problem(X[i]) if i in best_ids else y_pred[i] for i in range(len(X))]
            else:
                # If ratio==1 or no surrogate trained → evaluate full population
                X_real = X
                best_ids = np.arange(popsize)
                y_pred_chosen = None
                y = [problem(X[i]) if i in best_ids else y_pred[i] for i in range(len(X))]

            # True evaluation
            y_real = np.asarray(y)[best_ids].tolist()
            n_real_evals += len(y_real)

            # Update CMA-ES only using real data
            es.tell(X.tolist(), y)

            # Update surrogate
            if self.x_obs is None:
                self.x_obs = X_real
                self.y_obs = y_real
                self.ages = [it] * len(X_real)
            else:
                self.x_obs = np.vstack([self.x_obs, X_real])
                self.y_obs = np.concatenate([self.y_obs, y_real])
                self.ages = np.concatenate([self.ages, [it] * len(X_real)])

            #vis.update(self.x_obs, self.y_obs, self.ages)
            #input()

            # Inform RealEvaluationRatio
            if y_pred_chosen is not None:
                args.real_evaluation_ratio.update({
                    "true": y_real,
                    "pred": y_pred_chosen
                })

            best_so_far = np.min(self.y_obs)
            
            # Logging
            if L_dict is not None:
                model.update(self.x_obs, self.y_obs, best_so_far, L_dict)
                
            it += 1
            print(f"[SUR-CMA-ES Iter {it} Evals/dim {n_real_evals//dim}] f(best)={best_so_far:.5f}")

            if problem.final_target_hit:
                break
            
            if es.stop():
                break

        model.save()

        return problem.final_target_hit, n_real_evals/args.dimension
