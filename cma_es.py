from types import SimpleNamespace

import numpy as np
import cma
from scipy.linalg import expm

class CMAEvolutionStrategy:
    def __init__(self, x0, sigma0, options=None):
        self.options = options or {}
        
        if self.options.get("cma_type", "basic") == "basic":
            new_options = {k: v for k, v in self.options.items() if k != "cma_type"}
            self.impl = cma.CMAEvolutionStrategy(x0, sigma0, new_options)
            self.sm = SimpleNamespace(C=self.impl.sm.C)
        elif self.options.get("cma_type", "basic") == "led":            
            self.impl = CMALEDEvolutionStrategy(x0, sigma0, self.options)
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("cma_type", "basic") == "lm":
            self.impl = CMALMEvolutionStrategy(x0, sigma0, self.options)
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("cma_type", "basic") == "mo":            
            self.impl = CMAMOEvolutionStrategy(x0, sigma0, self.options)
            self.sm = SimpleNamespace(C=self.impl.C)
            
        self.sigma = self.impl.sigma
        self.mean = self.impl.mean

    def ask(self):
        ret_val = self.impl.ask()
        if self.options.get("cma_type", "basic") == "basic":
            self.sm = SimpleNamespace(C=self.impl.sm.C)
        elif self.options.get("cma_type", "basic") == "led":
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("cma_type", "basic") == "lm":
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("cma_type", "basic") == "mo":
            self.sm = SimpleNamespace(C=self.impl.C)
            
        self.sigma = self.impl.sigma
        self.mean = self.impl.mean
        return ret_val
        

    def tell(self, xs, ys):
        ret_val = self.impl.tell(xs, ys)
        if self.options.get("cma_type", "basic") == "basic":
            self.sm = SimpleNamespace(C=self.impl.sm.C)
        elif self.options.get("cma_type", "basic") == "led":
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("cma_type", "basic") == "lm":
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("cma_type", "basic") == "mo":
            self.sm = SimpleNamespace(C=self.impl.C)
            
        self.sigma = self.impl.sigma
        self.mean = self.impl.mean
        return ret_val

    def stop(self):
        return self.impl.stop()

    @property
    def result(self):
        return self.impl.result

class CMALEDEvolutionStrategy:
    class Result:
        def __init__(self):
            self.xbest = None
            self.fbest = np.inf

    def __init__(self, x0, sigma0, options=None):
        self.dim = len(x0)
        self.mean = np.array(x0, dtype=float)
        self.sigma = sigma0
        opts = options or {}
        self.lambda_ = opts.get("popsize", 4 + int(3 * np.log(self.dim)))
        self.mu = self.lambda_ // 2

        # --- full weight vector (length λ) ---
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.lambda_ + 1))
        self.weights = raw_weights[:self.mu]  # for mean update
        self.weights /= np.sum(self.weights)

        # Positive & negative normalized weights
        w_pos = np.maximum(raw_weights, 0)
        w_neg = np.minimum(raw_weights, 0)
        self.w_pos = w_pos / np.sum(w_pos)
        self.w_neg = w_neg / np.sum(np.abs(w_neg))

        # Effective sample sizes
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)
        self.mu_eff_pos = 1.0 / np.sum(self.w_pos ** 2)
        self.mu_eff_neg = 1.0 / np.sum(self.w_neg ** 2)

        # Learning rates
        self.cc = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mu_eff - 2 + 1 / self.mu_eff) /
                       ((self.dim + 2) ** 2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.cs

        # Evolution paths and covariance
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.C = np.eye(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.inv_sqrt_C = np.eye(self.dim)
        self.eigeneval = 0
        self.chiN = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

        # State
        self.arz = None
        self.ary = None
        self.arx = None
        self.result = self.Result()
        self.gen = 0

        # LED parameters
        self.q = opts.get("led_quantile", 0.5)
        self.alpha = opts.get("led_alpha", 0.2)
        self.beta = opts.get("led_beta", 0.5)
        self.snr_accum = np.zeros(self.dim)

        # Active option
        self.CMA_active = opts.get("CMA_active", False)

    def ask(self):
        self.arz = np.random.randn(self.lambda_, self.dim)
        self.ary = self.arz @ (self.B * self.D).T
        self.arx = self.mean + self.sigma * self.ary
        return self.arx

    def tell(self, solutions, fitnesses):
        self.gen += 1
        X = np.array(solutions)
        f = np.array(fitnesses)

        # Sort by fitness
        idx = np.argsort(f)
        X = X[idx]
        ary = self.ary[idx]
        f = f[idx]

        # Track best
        if f[0] < self.result.fbest:
            self.result.fbest = f[0]
            self.result.xbest = X[0].copy()

        # Recombination
        y_w = np.dot(self.weights, ary[:self.mu])
        self.mean += self.sigma * y_w

        # Evolution paths
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (self.inv_sqrt_C @ y_w)
        norm_ps = np.linalg.norm(self.ps)
        hsig = int(norm_ps / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chiN < (1.4 + 2 / (self.dim + 1)))
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y_w

        # LED SNR update
        snr_now = np.abs(y_w) / (np.sqrt(np.diag(self.C)) + 1e-12)
        self.snr_accum = (1 - self.alpha) * self.snr_accum + self.alpha * snr_now
        Neff = (np.sum(self.snr_accum) ** 2) / (np.sum(self.snr_accum ** 2) + 1e-12)
        ratio = Neff / self.dim
        phi = ratio ** self.beta

        # Scale learning rates
        c1 = self.c1 * phi
        cmu = self.cmu * phi
        cc = self.cc * phi
        cs = self.cs * phi

        # Select top-q dimensions
        threshold = np.quantile(self.snr_accum, 1 - self.q)
        mask = (self.snr_accum >= threshold)

        # Covariance update
        delta = (1 - hsig) * cc * (2 - cc)
        rank_one = c1 * (np.outer(self.pc, self.pc) + delta * self.C)

        # --- Active CMA-ES update ---
        if self.CMA_active:
            # Positive contributions (best)
            C_plus = cmu * (ary.T @ np.diag(self.w_pos) @ ary)

            # Negative contributions (worst)
            C_minus = cmu * (ary.T @ np.diag(np.abs(self.w_neg)) @ ary)

            # Scale factors
            c_mu_pos = min(1 - c1, cmu)
            c_mu_neg = min(1 - c1 - c_mu_pos,
                           (1 - c1 - c_mu_pos) * self.mu_eff_neg / (self.dim**2 + self.mu_eff_neg))

            self.C = (1 - c1 - c_mu_pos - c_mu_neg) * self.C + rank_one + c_mu_pos * C_plus - c_mu_neg * C_minus
        else:
            # Only positive contributions
            U = ary[:self.mu][:, mask]
            rank_mu = cmu * U.T @ np.diag(self.weights) @ U
            C_mu = np.zeros_like(self.C)
            C_mu[np.ix_(mask, mask)] = rank_mu
            self.C = (1 - c1 - cmu) * self.C + rank_one + C_mu

        # Step-size control
        self.sigma *= np.exp((cs / self.damps) * (norm_ps / self.chiN - 1))

        # Update eigen decomposition
        if (self.gen - self.eigeneval) > (self.lambda_ / (c1 + cmu) / self.dim / 10):
            self.eigeneval = self.gen
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            D2, B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(D2, 1e-30))
            self.B = B
            self.inv_sqrt_C = B @ np.diag(1.0 / self.D) @ B.T

    def stop(self):
        return self.sigma < 1e-12 or self.gen > 1e5

class CMALMEvolutionStrategy:
    class Result:
        def __init__(self):
            self.xbest = None
            self.fbest = np.inf

    def __init__(self, x0, sigma0, options=None):
        options = options or {}

        self.rng = np.random.default_rng(options.get("seed"))
        self.dim = len(x0)
        self.mean = np.array(x0, dtype=float)
        self.sigma = sigma0
        self.result = self.Result()

        # Population size
        self.lam = options.get("popsize", 4 + int(3 * np.log(self.dim)))
        self.mu = self.lam // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mueff = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)

        # Step-size control parameters
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.ds = 1 + self.cs + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1)
        self.cc = 4 / (self.dim + 4)

        # Learning rate for rank-one update
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)

        # Evolution paths
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        # Memory (u, v) pairs
        self.memory_size = options.get("memory_size", 4 + int(3 * np.log(self.dim)))
        self.U = []
        self.V = []
        self.C = np.eye(self.dim)

    # ----------------------------------------------------------------------
    # Utilities
    def apply_A(self, z):
        """Compute A * z via recursive application of stored (u, v)."""
        y = np.array(z, copy=True)
        for u, v in self.UV_pairs():
            y += u * np.dot(v, y)
        return y

    def apply_Ainv(self, z):
        y = np.array(z, copy=True)
        for u, v in reversed(list(zip(self.U, self.V))):
            y -= v * np.dot(u, y)
        return y

    def UV_pairs(self):
        return zip(self.U, self.V)

    # ----------------------------------------------------------------------
    def ask(self):
        """Sample a new population."""
        self.z = self.rng.normal(size=(self.lam, self.dim))
        self.y = np.array([self.apply_A(z) for z in self.z])
        self.x = self.mean + self.sigma * self.y
        return self.x

    def tell(self, solutions, fitnesses):
        """Update distribution parameters from evaluated solutions."""
        idx = np.argsort(fitnesses)
        x_sel = solutions[idx[:self.mu]]
        z_sel = self.z[idx[:self.mu]]

        # Save old mean
        old_mean = self.mean.copy()

        # Update mean
        self.mean = np.dot(self.weights, x_sel)

        # Update evolution paths
        zmean = np.dot(self.weights, z_sel)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * self.apply_Ainv(zmean)

        norm_ps = np.linalg.norm(self.ps)
        hsig = int(norm_ps / np.sqrt(1 - (1 - self.cs) ** 2) / np.sqrt(self.dim) < (1.4 + 2 / (self.dim + 1)))
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.mean - old_mean) / self.sigma

        # Store new (u, v) pair
        if hsig:
            v = self.apply_Ainv(self.pc)
            norm_v = np.linalg.norm(v)
            if norm_v > 0:
                v /= norm_v
                u = np.sqrt(self.c1) * self.pc
                if len(self.U) >= self.memory_size:
                    self.U.pop(0)
                    self.V.pop(0)
                self.U.append(u)
                self.V.append(v)

        C = np.eye(self.dim)
        for uu, vv in zip(self.U, self.V):
            d = np.dot(vv, uu)
            if abs(d) > 1e-20:
                add = np.outer(uu, vv) + np.outer(vv, uu)
                C += add / d
        self.C = 0.5 * (C + C.T)

        best_f = np.asarray(fitnesses)[np.argsort(fitnesses)][0]
        best_x = np.asarray(solutions)[np.argsort(fitnesses)][0]
        if best_f < self.result.fbest:
            self.result.fbest = best_f
            self.result.xbest = best_x.copy()

        # Step-size control
        self.sigma *= np.exp((self.cs / self.ds) * (norm_ps / np.sqrt(self.dim) - 1))

    def stop(self):
        return self.sigma < 1e-12

class CMAMOEvolutionStrategy:
    class Result:
        def __init__(self):
            self.xbest = None
            self.fbest = np.inf
            self.archive = []  # non-dominated archive

    def __init__(self, x0, sigma0, options=None):
        self.dim = len(x0)
        self.sigma = sigma0
        self.mean = np.array(x0, dtype=float)
        opts = options or {}

        self.num_strategies = opts.get("mo_num_strategies", 3)
        self.popsize = opts.get("popsize", 4 + int(3 * np.log(self.dim)))

        # initialize multiple CMA-ES strategies
        self.strategies = [
            cma.CMAEvolutionStrategy(x0, sigma0, {"popsize": self.popsize})
            for _ in range(self.num_strategies)
        ]

        # initialize C as identity
        self.C = np.eye(self.dim)

        self.result = self.Result()
        self.gen = 0

    # -------------------
    # ask / tell interface
    # -------------------
    def ask(self):
        """Generate offspring from all CMA strategies"""
        self.offspring = []
        self.origins = []
        for i, es in enumerate(self.strategies):
            xs = es.ask()
            self.offspring.extend(xs)
            self.origins.extend([i] * len(xs))
        return np.array(self.offspring)

    def tell(self, solutions, fitnesses):
        """
        fitnesses: array (N, m) of objective values
        """
        self.gen += 1
        X = np.array(solutions)
        F = np.array(fitnesses)

        # ensure 2D (N, m)
        if F.ndim == 1:
            F = F.reshape(-1, 1)

        origins = np.array(self.origins)

        # update Pareto archive
        self.result.archive = self.update_archive(self.result.archive, X, F)

        # non-dominated sorting
        ranks = self.fast_non_dominated_sort(F)
        distances = self.crowding_distance(F, ranks)

        # order offspring by Pareto rank + crowding distance
        order = sorted(range(len(X)), key=lambda i: (ranks[i], -distances[i]))
        X_sorted, F_sorted, origins_sorted = X[order], F[order], origins[order]

        # track best (via scalarization for logging)
        scalar_f = np.sum(F_sorted, axis=1)  # sum of objectives
        if scalar_f[0] < self.result.fbest:
            self.result.fbest = scalar_f[0]
            self.result.xbest = X_sorted[0].copy()

        # distribute back to each CMA
        for i, es in enumerate(self.strategies):
            idx = np.where(origins_sorted == i)[0]
            if len(idx) == 0:
                continue
            xs = X_sorted[idx]
            # scalarize multi-objective fitness for CMA update
            fs = np.sum(F_sorted[idx], axis=1)
            es.tell(xs.tolist(), fs.tolist())

        # update representative mean, sigma, C
        self.mean = np.mean([es.mean for es in self.strategies], axis=0)
        self.sigma = np.mean([es.sigma for es in self.strategies])
        self.C = np.mean([es.C for es in self.strategies], axis=0)

    # -------------------
    # Pareto archive
    # -------------------
    def update_archive(self, archive, X, F):
        if not archive:
            all_X, all_F = X, F
        else:
            old_X, old_F = zip(*archive)
            all_X = np.vstack([np.array(old_X), X])
            all_F = np.vstack([np.array(old_F), F])
        mask = self.non_dominated_mask(all_F)
        return [(all_X[i], all_F[i]) for i in range(len(all_X)) if mask[i]]

    @staticmethod
    def dominates(f1, f2):
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def non_dominated_mask(self, F):
        n = len(F)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if i != j and self.dominates(F[j], F[i]):
                    mask[i] = False
                    break
        return mask

    def fast_non_dominated_sort(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(n):
                if self.dominates(F[j], F[i]):
                    ranks[i] += 1
        return ranks

    def crowding_distance(self, F, ranks):
        n, m = F.shape
        distances = np.zeros(n)
        fronts = {}
        for i, r in enumerate(ranks):
            fronts.setdefault(r, []).append(i)
        for front in fronts.values():
            front_F = F[front]
            for k in range(m):
                idx = np.argsort(front_F[:, k])
                f_min, f_max = front_F[idx[0], k], front_F[idx[-1], k]
                distances[front[idx[0]]] = distances[front[idx[-1]]] = np.inf
                if f_max > f_min:
                    for j in range(1, len(idx) - 1):
                        d = (front_F[idx[j + 1], k] - front_F[idx[j - 1], k]) / (f_max - f_min)
                        distances[front[idx[j]]] += d
        return distances

    # -------------------
    # stopping / results
    # -------------------
    def stop(self):
        return any(es.stop() for es in self.strategies)

    def pareto_front(self):
        return self.result.archive
