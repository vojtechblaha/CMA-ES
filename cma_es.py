from types import SimpleNamespace

import numpy as np
import cma
from scipy.linalg import expm, norm, solve
from types import SimpleNamespace
from scipy.spatial.distance import cdist
from scipy.stats import chi

class CMAEvolutionStrategy:
    def __init__(self, x0, sigma0, options=None):
        self.options = options or {}

        if self.options.get("optimizer", "cma") == "cma":
            if self.options.get("cma_type", "basic") == "basic":
                new_options = {k: v for k, v in self.options.items() if (k not in [
                    "cma_type", "func", "optimizer", "slsqp_ftol",
                    "slsqp_difference_gradient", "hees_tol", "shade_lm"])}
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

        elif self.options.get("optimizer", "cma") == "slsqp":
            self.impl = SLSQP(x0, sigma0, self.options)
            self.sm = SimpleNamespace(C=self.impl.C)

        elif self.options.get("optimizer", "cma") == "fmincon":
            self.impl = FMinCon(x0, sigma0, self.options)
            self.sm = SimpleNamespace(C=self.impl.C)

        elif self.options.get("optimizer", "cma") == "fminunc":
            self.impl = FMinUnc(x0, sigma0, self.options)
            self.sm = SimpleNamespace(C=self.impl.C)

        elif self.options.get("optimizer", "cma") == "csl":
            self.impl = CSL(x0, sigma0, self.options)
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("optimizer", "cma") == "hees":
            self.impl = HEES(x0, sigma0, self.options)
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("optimizer", "cma") == "shade":
            self.impl = SHADE(x0, sigma0, self.options)
            self.sm = SimpleNamespace(C=self.impl.C)

        self.sigma = self.impl.sigma
        self.mean = self.impl.mean
            

    def ask(self):
        ret_val = self.impl.ask()
        if self.options.get("optimizer", "cma") == "cma":
            if self.options.get("cma_type", "basic") == "basic":
                self.sm = SimpleNamespace(C=self.impl.sm.C)
            elif self.options.get("cma_type", "basic") == "led":
                self.sm = SimpleNamespace(C=self.impl.C)
            elif self.options.get("cma_type", "basic") == "lm":
                self.sm = SimpleNamespace(C=self.impl.C)
            elif self.options.get("cma_type", "basic") == "mo":
                self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("optimizer", "cma") == "slsqp":
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("optimizer", "cma") == "fmincon":
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("optimizer", "cma") == "fminunc":
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("optimizer", "cma") == "csl":
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("optimizer", "cma") == "hees":
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("optimizer", "cma") == "shade":
            self.sm = SimpleNamespace(C=self.impl.C)
    
        self.sigma = self.impl.sigma
        self.mean = self.impl.mean
        return ret_val
        

    def tell(self, xs, ys, xs_inv_idx = None, xs_len = None):
        if self.options.get("optimizer", "cma") in ["hees", "shade"]:
            ret_val = self.impl.tell(xs, ys, xs_inv_idx, xs_len)
        else:
            ret_val = self.impl.tell(xs, ys)
        if self.options.get("optimizer", "cma") == "cma":
            if self.options.get("cma_type", "basic") == "basic":
                self.sm = SimpleNamespace(C=self.impl.sm.C)
            elif self.options.get("cma_type", "basic") == "led":
                self.sm = SimpleNamespace(C=self.impl.C)
            elif self.options.get("cma_type", "basic") == "lm":
                self.sm = SimpleNamespace(C=self.impl.C)
            elif self.options.get("cma_type", "basic") == "mo":
                self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("optimizer", "cma") == "slsqp":
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("optimizer", "cma") == "fmincon":
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("optimizer", "cma") == "fminunc":
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("optimizer", "cma") == "csl":
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("optimizer", "cma") == "hees":
            self.sm = SimpleNamespace(C=self.impl.C)
        elif self.options.get("optimizer", "cma") == "shade":
            self.sm = SimpleNamespace(C=self.impl.C)
            
        self.sigma = self.impl.sigma
        self.mean = self.impl.mean
        return ret_val

    def stop(self):
        return self.impl.stop()

    @property
    def result(self):
        return self.impl.result

class SHADE:
    """
    Exact algorithmic reimplementation of SLSQP (Sequential Least Squares Programming)
    as used in SciPy 1.2.1 (SLSQP-scipy-2019 benchmark in GECCO paper).
    """
    class Result:
        def __init__(self):
            self.xbest = None
            self.fbest = np.inf

    def __init__(self, x0, sigma0, options=None):
        self.options = options or {}
        self.dim = len(x0)
        self.bounds = np.asarray([(-5, 5)] * self.dim)
        self.shade_lm = self.options.get("shade_lm", None)
        self.func = self.options.get("func", None)
        self.pop_size = options.get("popsize", 4 + int(3 * np.log(self.dim)))
        self.model_use_count = int(0.05 * self.pop_size)

        # SHADE parameters
        self.stagnation = 0
        self.H_idx = 0
        self.H = 11
        self.CR_m = np.full(self.H, 0.9)
        self.F_m = np.full(self.H, 0.38)
        self.p_best_rate = 0.11
        self.archive = []
        self.archive_size = self.pop_size
        
        self.iter = 0
        self.lm_interval = 20

        # Restart parameters
        self.no_improve_limit = 5000 * self.dim
        self.val_tol = 1e-12
        self.loc_tol = 1e-12
        
        self.pop, self.fit = self._init_population()

        self.mean = np.asarray(self.pop[np.argmin(self.fit)], dtype=float).copy()
        self.sigma = sigma0
        self.C = np.eye(self.dim)
        self.best_x = self.mean.copy()
        self.best_f = np.min(self.fit)
        self.result = self.Result()

    def _init_population(self):
        pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.pop_size, self.dim))
        fit = np.array([self.func(x) for x in pop])
        return pop, fit

    def _fit_quadratic_model(self, X, y):
        # Fit full quadratic model: f(x) = xᵀAx + bᵀx + c
        n = X.shape[0]
        A = np.zeros((n, self.dim**2 + self.dim + 1))
        for i in range(n):
            xi = X[i]
            quad = np.outer(xi, xi)[np.triu_indices(self.D)]
            A[i] = np.concatenate([quad, xi, [1]])
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
        return coeffs

    def _stationary_point(self, coeffs):
        # Compute stationary point from ∇f(x)=0: 2A_sym x + b = 0
        n_quad = self.dim * (self.dim + 1) // 2
        a_flat = coeffs[:n_quad]
        b = coeffs[n_quad:n_quad + self.dim]
        A = np.zeros((self.dim, self.dim))
        tri = np.triu_indices(self.dim)
        A[tri] = a_flat
        A = A + np.triu(A, 1).T
        try:
            x_theta = np.linalg.solve(2 * A, -b)
        except np.linalg.LinAlgError:
            x_theta = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
        return np.clip(x_theta, self.bounds[:, 0], self.bounds[:, 1])

    def _mutation(self, pop, fit, idx, H_idx):
        F = np.clip(np.random.standard_cauchy() * 0.1 + self.F_m[H_idx], 0, 1)
        p_num = max(2, int(self.p_best_rate * len(pop)))
        x_pbest = pop[np.random.choice(np.argsort(fit)[:p_num])]
        r1, r2 = np.random.choice(len(pop), 2, replace=False)
        x_r1 = pop[r1]
        x_r2 = self.archive[np.random.randint(len(self.archive))] if self.archive else pop[r2]
        v = pop[idx] + F * (x_pbest - pop[idx]) + F * (x_r1 - x_r2)
        return np.clip(v, self.bounds[:,0], self.bounds[:,1]), F

    def _crossover(self, x, v, h_idx):
        CR = np.clip(np.random.normal(self.CR_m[h_idx], 0.1), 0, 1)
        mask = np.random.rand(self.dim) < CR
        mask[np.random.randint(self.dim)] = True
        u = np.where(mask, v, x)
        return np.clip(u, self.bounds[:,0], self.bounds[:,1]), CR

    def _update_memory(self, success_F, success_CR, df, H_idx):
        if success_F:
            w = df / np.sum(df)
            self.F_m[H_idx] = np.sum(w * np.array(success_F) ** 2) / np.sum(w * np.array(success_F))
            self.CR_m[H_idx] = np.sum(w * np.array(success_CR))

    def _fit_model(self, X, y):
        n = len(X)
        Phi = np.zeros((n, int(self.dim*(self.dim+1)/2) + self.dim + 1))
        for i, xi in enumerate(X):
            Phi[i,:] = np.concatenate([xi[np.triu_indices(self.dim,0, self.dim)].ravel()
                                       if self.dim==1 else np.outer(xi,xi)[np.triu_indices(self.dim)],
                                       xi, [1]])
        coeffs, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        n_quad = self.dim*(self.dim+1)//2
        a_flat = coeffs[:n_quad]
        b = coeffs[n_quad:n_quad+self.dim]
        c = coeffs[-1]
        H = np.zeros((self.dim,self.dim))
        tri = np.triu_indices(self.dim)
        H[tri] = a_flat
        H = H + np.triu(H,1).T
        g = b
        return H, g, c

    def _lm_step(self, x0, H, g, lam):
        """One LM step with gain-ratio control."""
        I = np.eye(self.dim)
        try:
            dx = np.linalg.solve(H + lam * I, -g)
        except np.linalg.LinAlgError:
            dx = -np.linalg.pinv(H + lam * I) @ g
        x_new = np.clip(x0 + dx, self.bounds[:,0], self.bounds[:,1])
        f_old = self.func(x0)
        f_new = self.func(x_new)
        num = f_old - f_new
        denom = -0.5 * dx @ (g + H @ dx)
        rho = num / abs(denom) if denom != 0 else 0
        if rho > 0:
            lam /= 10
            x_best, f_best = x_new, f_new
        else:
            lam *= 10
            x_best, f_best = x0, f_old
        return x_best, f_best, lam

    def _local_model_search(self, pop, fit):
        best_idx = np.argmin(fit)
        x_best = pop[best_idx]
        r = 0.25 * np.sqrt(self.dim) * np.mean(self.bounds[:,1]-self.bounds[:,0])
        dist = np.linalg.norm(pop - x_best, axis=1)
        idx = np.argsort(dist)[:2*self.dim+1]
        X_sel, y_sel = pop[idx], fit[idx]
        H, g, _ = self._fit_model(X_sel, y_sel)
        x = x_best.copy()
        lam = 1e-3
        for _ in range(10):
            x, f, lam = self._lm_step(x, H, g, lam)
        return x, f

    # ---------------------------------------------------------------------
    def ask(self):
        """Return the current iterate x (CMA-compatible interface)."""
        # --- SHADE loop ---
        self.iter += 1
        self.Fs, self.CRs, xs = [], [], []
        for i in range(self.pop_size):
            v_i, F_i = self._mutation(self.pop, self.fit, i, self.H_idx)
            u_i, CR_i = self._crossover(self.pop[i], v_i, self.H_idx)
            xs.append(u_i)
            self.Fs.append(F_i)
            self.CRs.append(CR_i)

        return xs

    # ---------------------------------------------------------------------
    def tell(self, xs, ys, xs_inv_idx, xs_len):
        """
        Receives f(x) and updates the internal state.
        This is the 'external evaluation' interface.
        """
        xs = np.asarray(xs).copy()[xs_inv_idx][-xs_len:]
        ys = np.asarray(ys).copy()[xs_inv_idx][-xs_len:]

        new_pop, new_fit = np.copy(self.pop), np.copy(self.fit)
        success_F, success_CR, df = [], [], []
        for i in range(self.pop_size):
            f_u = ys[i]
            if f_u < self.fit[i]:
                if len(self.archive) >= self.archive_size:
                    self.archive.pop(np.random.randint(len(self.archive)))
                self.archive.append(self.pop[i].copy())
                new_pop[i], new_fit[i] = xs[i], ys[i]
                success_F.append(self.Fs[i])
                success_CR.append(self.CRs[i])
                df.append(abs(self.fit[i] - ys[i]))

        self.pop, self.fit = new_pop, new_fit
        self._update_memory(success_F, success_CR, np.array(df), self.H_idx)
        self.H_idx = (self.H_idx + 1) % self.H
        
        if np.min(ys) < self.best_f:
            self.best_f = np.min(ys)
            self.best_x = xs[np.argmin(ys)].copy()
            self.stagnation = 0
        else:
            self.stagnation += 1

        # Lokální model každých 0.05 * NP evaluací
        if self.shade_lm and self.iter % self.lm_interval == 0:
            x_lm, f_lm = self._local_model_search(self.pop, self.fit)
            worst = np.argmax(self.fit)
            if f_lm < self.fit[worst]:
                self.pop[worst], self.fit[worst] = x_lm, f_lm
                if f_lm < self.best_f:
                    self.best_f = f_lm
                    self.best_x = x_lm
                    stagnation = 0

        self.result.fbest = self.best_f
        self.result.xbest = self.best_x
        self.mean = self.best_x

        # Restart conditions
        if self.stagnation > self.no_improve_limit or \
            (np.max(self.fit) - np.min(self.fit) < self.val_tol) or \
            np.max(np.ptp(self.pop, axis=0)) < self.loc_tol:
            self.pop, self.fit = self._init_population()
            self.stagnation = 0
            self.archive.clear()

    # ---------------------------------------------------------------------
    def stop(self):
        """Stopping criterion (CMA-compatible)."""
        return self.stagnation >= self.no_improve_limit

class StepND:
    class Step1D:
        def __init__(self, f, a, b, f_star=None, tol=1e-8):
            self.f = f
            self.a, self.b = float(a), float(b)
            self.points = [self.a, self.b]
            self.values = [f(self.a), f(self.b)]
            self.f_star = np.min(self.values) if f_star is None else f_star
            self.l = tol

        def _difficulty(self, i):
            x1, x2 = self.points[i], self.points[i+1]
            f1, f2 = self.values[i], self.values[i+1]
            dx = x2 - x1
            dy = f2 - f1
            y_hat = self.f_star - f1 + self.l
            val = y_hat**2 - y_hat * dy
            if val < 0:
                val = 0.0
            return (4*y_hat - 2*dy + 4*np.sqrt(val)) / (dx**2)

        def step(self):
            order = np.argsort(self.points)
            self.points = list(np.array(self.points)[order])
            self.values = list(np.array(self.values)[order])
            D_vals = [self._difficulty(i) for i in range(len(self.points)-1)]
            best_i = int(np.argmin(D_vals))
            x_new = 0.5 * (self.points[best_i] + self.points[best_i+1])
            f_new = self.f(x_new)
            self.points.insert(best_i+1, x_new)
            self.values.insert(best_i+1, f_new)
            if f_new < self.f_star:
                self.f_star = f_new
            return x_new, f_new
    
    class Result:
        def __init__(self):
            self.xbest = None
            self.fbest = np.inf
            
    def __init__(self, f, x0, step_size, tol):
        self.f = f
        self.tol = tol
        self.size = step_size
        self.n = len(x0)
        self.iters = 0

        self.best_x = np.asarray(x0, dtype=float).copy()
        self.best_f = np.inf
        self.result = self.Result()
        
        self.lines = []
        for i in range(self.n):
            a, b = x0[i] - self.size, x0[i] + self.size
            fi = lambda xi, i=i: self.f(self._combine(i, xi))
            self.lines.append(StepND.Step1D(fi, a, b, f_star=self.best_f, tol=self.tol))

    def _combine(self, i, xi):
        x = np.copy(self.best_x)
        x[i] = xi
        return x

    def ask(self):
        self.iters += 1
        self.new_coords = []
        for i, line in enumerate(self.lines):
            xi_new, fi_new = line.step()
            best_idx = int(np.argmin(line.values))
            self.new_coords.append(line.points[best_idx])
        return np.array([self.new_coords])

    def tell(self, xs, ys):
        f_new = ys[0]
        improved = f_new + self.tol < self.best_f
        if improved:
            self.best_x = self.new_coords
            self.best_f = f_new
            for line in self.lines:
                line.f_star = self.best_f
        return self.best_x, self.best_f, improved
        

class HEES:
    """
    Exact algorithmic reimplementation of SLSQP (Sequential Least Squares Programming)
    as used in SciPy 1.2.1 (SLSQP-scipy-2019 benchmark in GECCO paper).
    """
    class Result:
        def __init__(self):
            self.xbest = None
            self.fbest = np.inf

    def __init__(self, x0, sigma0, options=None):
        self.options = options or {}
        self.tol = self.options.get("hees_tol", 1e-10)
        self.f = self.options.get("func", None)
        self.d = len(x0)
        self.rng = np.random.default_rng(42)
        
        self.lam_tilde = 2 + int(3/2 * np.log(self.d))
        self.lam = 2 * self.lam_tilde
        self.B = int(np.ceil(self.lam_tilde / self.d))

        self.w = self._cma_weights(self.lam)
        self.mu_eff = 1.0 / np.sum(self.w**2)
        
        self.cs = (self.mu_eff + 2) / (self.d + self.mu_eff + 5)
        self.ds = 1 + 2 * max(0.0, np.sqrt((self.mu_eff - 1) / (self.d + 1)) - 1) + self.cs
        
        self.mu_eff_mirrored = self.mu_eff / (1.0 - (self.mu_eff - 1.0) / (2*self.lam_tilde - 1.0))
        
        self.mean = np.zeros(self.d)
        self.A = np.eye(self.d)
        self.sigma = sigma0
        self.ps = np.zeros(self.d)
        self.gs = 0.0

        self.chiN = np.sqrt(self.d) * (1 - 1/(4*self.d) + 1/(21*self.d**2))

        self.bij_blocks, self.bij_list = [], []

        self.C = np.eye(self.d)
        self.best_x = np.asarray(x0, dtype=float).copy()
        self.best_f = np.inf
        self.result = self.Result()

    def _cma_weights(self, lam):
        w = np.log(lam / 2.0 + 0.5) - np.log(np.arange(1, lam + 1))
        w = np.maximum(w, 0)
        w /= np.sum(w)
        return w

    def _sample_orthogonal(self):
        z = np.random.randn(self.d, self.d)
        n = np.linalg.norm(z, axis=0)
        # Gram–Schmidt (QR)
        q, _ = np.linalg.qr(z)
        for i in range(self.d):
            q[:, i] *= n[i]
        return q.T  # returns list of d vectors (rows)

    def _compute_G(self, B, d, lam_tilde, bij_blocks, f_m, f_xpm_flat, sigma, kappa=3.0, etaA=0.5):
        """
        bij_blocks: list B bloků, každý (d, d) – řádky jsou b_ij
        f_xpm_flat: list délky λ̃, prvky (f_plus, f_minus) pro prvních λ̃ směrů
        """
        total_dirs = B * d
        h = np.zeros(total_dirs)
        used = 0

        # curvature jen pro prvních λ̃ směrů
        for bi in range(B):
            block = bij_blocks[bi]
            for dj in range(d):
                flat_index = bi * d + dj
                if used < lam_tilde:
                    b = block[dj]
                    f_plus, f_minus = f_xpm_flat[used]
                    h[flat_index] = (f_plus + f_minus - 2 * f_m) / (sigma**2 * norm(b)**2)
                    used += 1
                else:
                    # zbytek zůstává 0, q tam pak nastavíme na 0 (neutral)
                    pass

        # pokud jsou všechny odhady <= 0, vrátíme identitu
        used_mask = np.zeros_like(h, dtype=bool)
        used_mask[:lam_tilde] = True

        if np.max(h[used_mask]) <= 0:
            return np.eye(d)

        max_h = np.max(h[used_mask])
        c = max_h / kappa

        # clipping jen pro použité směry
        h_clipped = h.copy()
        h_clipped[used_mask] = np.maximum(h[used_mask], c)

        q = np.zeros_like(h_clipped)
        q[used_mask] = np.log(h_clipped[used_mask])

        # odečti průměr (jen z použitých) → jednotkový determinant
        mean_q = np.mean(q[used_mask])
        q[used_mask] -= mean_q

        # learning rate a exponent -1/2
        q *= -etaA / 2.0

        # neutrální update ve směrech, kde jsme neměli curvature (zbytek)
        for idx in range(lam_tilde, total_dirs):
            q[idx] = 0.0

        # G = 1/B * sum_ij exp(q_ij)/||b_ij||^2 * b_ij b_ij^T
        G = np.zeros((d, d))
        flat_index = 0
        for bi in range(B):
            block = bij_blocks[bi]
            for dj in range(d):
                b = block[dj]
                b_vec = b.reshape(-1, 1)
                G += np.exp(q[flat_index]) / (norm(b)**2) * (b_vec @ b_vec.T)
                flat_index += 1
        G /= B

        return G

    # ---------------------------------------------------------------------
    def ask(self):
        """Return the current iterate x (CMA-compatible interface)."""
        self.bij_blocks, self.bij_list, xs = [], [], []
        
        for _ in range(self.B):
            block = self._sample_orthogonal()
            self.bij_blocks.append(block)
            for b in block:
                x_plus = self.mean + self.sigma * self.A @ b
                x_minus = self.mean - self.sigma * self.A @ b
                xs.append(x_plus)
                xs.append(x_minus)
                self.bij_list.append(b)
                if len(xs) // 2 >= self.lam_tilde:
                    break
            if len(xs) // 2 >= self.lam_tilde:
                break
        return xs

    # ---------------------------------------------------------------------
    def tell(self, xs, ys, xs_inv_idx, xs_len):
        """
        Receives f(x) and updates the internal state.
        This is the 'external evaluation' interface.
        """
        xs = np.asarray(xs).copy()[xs_inv_idx][-xs_len:]
        ys = np.asarray(ys).copy()[xs_inv_idx][-xs_len:]

        X_plus, X_minus, F_plus, F_minus, f_xpm = [], [], [], [], []
        for i in range(0, len(xs), 2):
            X_plus.append(xs[i])
            X_minus.append(xs[i+1])
            F_plus.append(ys[i])
            F_minus.append(ys[i+1])
            f_xpm.append((ys[i], ys[i+1]))

        f_m = self.f(self.mean)
        if f_m < self.best_f:
            self.best_f = f_m
            self.best_x = self.mean.copy()
            
        G = self._compute_G(self.B, self.d, self.lam_tilde, self.bij_blocks, f_m, f_xpm, self.sigma)
        self.A = self.A @ G

        X_all = np.vstack([X_plus, X_minus])
        F_all = np.hstack([F_plus, F_minus])
        
        idx_sorted = np.argsort(F_all)
        ranks = np.empty_like(idx_sorted)
        ranks[idx_sorted] = np.arange(len(F_all))
        w_all = self.w[ranks]
        
        self.mean = np.sum(w_all[:, None] * X_all, axis=0)

        # cumulative step-size adaptation
        w_plus = w_all[:self.lam_tilde]
        w_minus = w_all[self.lam_tilde:]
        delta_b = np.zeros(self.d)
        for i in range(self.lam_tilde):
            delta_b += (w_plus[i] - w_minus[i]) * self.bij_list[i]
        
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff_mirrored) * delta_b
        self.gs = (1 - self.cs)**2 * self.gs + self.cs * (2 - self.cs)

        self.sigma *= np.exp((self.cs / self.ds) * (norm(self.ps) / self.chiN - np.sqrt(self.gs)))

        
        if np.min(ys) < self.best_f:
            self.best_f = np.min(ys)
            self.best_x = xs[np.argmin(ys)].copy()

        self.result.fbest = self.best_f
        self.result.xbest = self.best_x

    # ---------------------------------------------------------------------
    def stop(self):
        """Stopping criterion (CMA-compatible)."""
        return self.sigma < self.tol

class CSL:
    class Result:
        def __init__(self):
            self.xbest = None
            self.fbest = np.inf

    def __init__(self, x0, sigma0, options=None):
        self.options = options
        self.dim = len(x0)
        self.bounds = np.asarray([(-5, 5)] * self.dim)
        self.lower, self.upper = self.bounds[:, 0], self.bounds[:, 1]
        self.func = self.options.get("func", None)
        self.n_init = int(self.options.get("n_init", 10))
        self.batch_size = int(self.options.get("batch_size", 10))
        self.rho = self.options.get("rho", 0.5)
        self.not_improving = 0
        
        self.C = np.eye(self.dim)
        self.mean = np.array(x0, dtype=float)
        self.sigma = sigma0
        self.result = self.Result()

        self.X, self.y = self._sample(self.n_init)

    # ------------------------------------------------------------------
    def _sample(self, n):
        """Uniform random sampling in the search domain."""
        x = self.lower + np.random.rand(n, self.dim) * (self.upper - self.lower)
        f = np.apply_along_axis(self.func, 1, x)
        return x, f

    # ------------------------------------------------------------------
    def _cluster_single_linkage(self, X, f):
        """
        Perform single-linkage clustering.
        Only the best point of each cluster is selected as a start for local search.
        """
        n = len(X)
        dists = cdist(X, X)
        np.fill_diagonal(dists, np.inf)
        radius = self.rho * np.mean(self.upper - self.lower) / (n ** (1.0 / self.dim))

        visited = np.zeros(n, dtype=bool)
        clusters = []
        for i in np.argsort(f):
            if visited[i]:
                continue
            cluster_indices = [i]
            for j in range(n):
                if not visited[j] and dists[i, j] < radius:
                    cluster_indices.append(j)
                    visited[j] = True
            clusters.append(cluster_indices)
        # Return the best representative of each cluster
        reps = [cluster[np.argmin(f[cluster])] for cluster in clusters]
        return [X[i] for i in reps]

    def ask(self):
        return self._cluster_single_linkage(np.array(self.X), np.array(self.y))

    def tell(self, xs, ys):
        X_new, y_new = self._sample(self.batch_size)
        self.X = np.vstack([self.X, X_new])
        self.y = np.concatenate([self.y, y_new])

        if abs(self.result.fbest - np.min(self.y)) < 1e-12:
            self.not_improving += 1
        else:
            self.not_improving = 0

        self.result.xbest = self.X[np.argmin(self.y)]
        self.result.fbest = np.min(self.y)

    def stop(self):
        return self.sigma < 1e-12 or self.not_improving > 10

class FMinCon:
    """
    Exact algorithmic reimplementation of MATLAB fmincon (interior-point)
    as used in Pál (2013): "Comparison of Multistart Global Optimization Algorithms
    on the BBOB Noiseless Testbed", GECCO 2013.

    This class mirrors the SLSQP-style structure for CMA-compatible interfacing.
    """

    class Result:
        def __init__(self):
            self.xbest = None
            self.fbest = np.inf

    # ---------------------------------------------------------------------
    def __init__(self, x0, sigma0, options=None):
        self.options = options or {}
        self.x = np.asarray(x0, dtype=float)
        self.func = self.options.get("func", None)
        self.grad = self.options.get("grad", None)
        self.constraints = self.options.get("constraints", None)  # list of callables g_i(x) >= 0
        self.bounds = self.options.get("bounds", None)
        self.A_ineq = self.options.get("A_ineq", None)
        self.b_ineq = self.options.get("b_ineq", None)
        self.A_eq = self.options.get("A_eq", None)
        self.b_eq = self.options.get("b_eq", None)
        self.dim = len(x0)

        # Algorithmic constants (matching fmincon settings in Pál 2013)
        self.ftol = self.options.get("ftol", 1e-12)
        self.ctol = self.options.get("ctol", 1e-12)
        self.mu_init = self.options.get("mu_init", 1.0)
        self.mu_decay = self.options.get("mu_decay", 0.5)
        self.alpha_init = self.options.get("alpha", 1.0)
        self.max_iter = int(self.options.get("max_iter", 1e8 * self.dim))
        self.max_eval = int(self.options.get("max_eval", 1e8 * self.dim))
        self.verbose = self.options.get("verbose", False)

        # Internal state
        self.iter = 0
        self.mu = self.mu_init
        self.B = np.eye(self.dim) * 0.01
        self.f = None
        self.g = None
        self.best_x = self.x.copy()
        self.best_f = np.inf
        self.converged = False
        self.nfev = 0
        self.result = self.Result()

        # Multipliers and slacks
        self.lambda_ = None  # nonlinear + linear ineq multipliers
        self.s = None        # slack variables

        self.C = np.eye(self.dim)
        self.mean = np.array(x0, dtype=float)
        self.sigma = sigma0

    # ---------------------------------------------------------------------
    def _evaluate(self, x):
        """Evaluate objective and gradient (finite differences if not provided)."""
        self.nfev += 1
        f = self.func(x)

        # Box bounds via log-barrier (interior-point)
        if self.bounds is not None:
            for i, (lb, ub) in enumerate(self.bounds):
                if lb is not None:
                    if x[i] <= lb:
                        return np.inf, np.zeros_like(x)
                    f -= self.mu * np.log(x[i] - lb)
                if ub is not None:
                    if x[i] >= ub:
                        return np.inf, np.zeros_like(x)
                    f -= self.mu * np.log(ub - x[i])

        if self.grad is not None:
            g = self.grad(x)
        else:
            eps = 1e-8
            g = np.zeros_like(x)
            fx = f
            for i in range(self.dim):
                x_step = x.copy()
                x_step[i] += eps
                g[i] = (self.func(x_step) - fx) / eps
        return f, g

    # ---------------------------------------------------------------------
    def _constraints_eval(self, x):
        """Evaluate all constraints and their Jacobians."""
        cons_vals, cons_jac = [], []

        # Nonlinear inequality constraints g(x) >= 0
        if self.constraints is not None:
            eps = 1e-8
            for gfun in self.constraints:
                val = gfun(x)
                grad = np.zeros(self.dim)
                for j in range(self.dim):
                    x_step = x.copy()
                    x_step[j] += eps
                    grad[j] = (gfun(x_step) - val) / eps
                cons_vals.append(val)
                cons_jac.append(grad)

        # Linear inequality constraints A_ineq x <= b_ineq -> g = b_ineq - A_ineq x >= 0
        if self.A_ineq is not None:
            g_lin = self.b_ineq - self.A_ineq @ x
            cons_vals.extend(g_lin)
            cons_jac.extend([-row for row in self.A_ineq])

        c = np.array(cons_vals)
        J = np.array(cons_jac)
        return c, J

    # ---------------------------------------------------------------------
    def _eq_constraints(self, x):
        """Evaluate linear equalities A_eq x = b_eq."""
        if self.A_eq is None:
            return np.array([]), np.zeros((0, self.dim))
        val = self.A_eq @ x - self.b_eq
        return val, self.A_eq

    # ---------------------------------------------------------------------
    def ask(self):
        return [self.x.copy()]

    # ---------------------------------------------------------------------
    def tell(self, xs, ys):
        x = xs[0]
        f = ys[0]
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()

        self._step()
        self.result.xbest = self.best_x
        self.result.fbest = self.best_f

        self.mean = self.x.copy()

    # ---------------------------------------------------------------------
    def _step(self):
        """Perform one primal-dual interior-point Newton–KKT step."""
        self.iter += 1
        f, grad_f = self._evaluate(self.x)
        c, J = self._constraints_eval(self.x)
        ceq, Aeq = self._eq_constraints(self.x)
        m = len(c)
        meq = len(ceq)

        # Initialize multipliers/slacks if first iteration
        if self.lambda_ is None:
            self.lambda_ = np.ones(m)
            self.s = np.maximum(1e-3, c.copy())

        # Build KKT system:
        # [ B   J^T  Aeq^T  0  ][dx]   = [ -grad_f - J^T λ - Aeq^T ν ]
        # [ J    0     0    I  ][dλ]   = [ -c - s                  ]
        # [ Aeq  0     0    0  ][dν]   = [ -ceq                    ]
        # [ 0  diag(λ) 0  diag(s) ][ds] = [ -λ*s + μe              ]

        diag_lambda = np.diag(self.lambda_)
        diag_s = np.diag(self.s)

        # Dual, primal, centering, equality residuals
        nu = np.zeros(meq)
        r_dual = -(grad_f + J.T @ self.lambda_ + (Aeq.T @ nu if meq > 0 else 0))
        r_primal = -(c + self.s)
        r_cent = -(self.lambda_ * self.s - self.mu * np.ones(m))
        r_eq = -ceq if meq > 0 else np.array([])

        if len(c) == 0:
            J = np.zeros((0, self.dim))
            self.lambda_ = np.zeros(0)
            self.s = np.zeros(0)

        # Construct full KKT system
        KKT = np.block([
            [self.B, J.T, Aeq.T if meq > 0 else np.zeros((self.dim, 0)), np.zeros((self.dim, m))],
            [J, np.zeros((m, m)), np.zeros((m, meq)), np.eye(m)],
            [Aeq if meq > 0 else np.zeros((0, self.dim)), np.zeros((meq, m)), np.zeros((meq, meq)), np.zeros((meq, m))],
            [np.zeros((m, self.dim)), diag_lambda, np.zeros((m, meq)), diag_s]
        ])
        rhs = np.concatenate([r_dual, r_primal, r_eq, r_cent])

        try:
            step = solve(KKT, rhs)
        except np.linalg.LinAlgError:
            if self.verbose:
                print("⚠️ Singular KKT system, fallback gradient step.")
            step = np.concatenate([-grad_f, np.zeros(KKT.shape[0] - len(grad_f))])

        dx = step[:self.dim]

        # Dampen if step is too large
        max_step = 0.5 * np.sqrt(np.dot(self.x, self.x) + 1e-8)
        if norm(dx) > max_step:
            dx *= max_step / (norm(dx) + 1e-12)
            
        dλ = step[self.dim:self.dim + m]
        dν = step[self.dim + m:self.dim + m + meq] if meq > 0 else np.zeros(0)
        ds = step[-m:]

        # Line search: maintain λ > 0, s > 0
        alpha = self.alpha_init

        # Line search: only if inequality constraints exist
        if len(self.s) > 0 and len(ds) > 0:
            while np.any(self.s + alpha * ds <= 0) or np.any(self.lambda_ + alpha * dλ <= 0):
                alpha *= 0.5
                if alpha < 1e-8:
                    break

        # Update primal variables
        self.x += alpha * dx

        # Update inequality-related variables only if present
        if len(self.s) > 0:
            self.lambda_ += alpha * dλ
            self.s += alpha * ds

        # BFGS update of Hessian
        f_new, grad_new = self._evaluate(self.x)
        y = grad_new - grad_f
        s_vec = alpha * dx
        ys = np.dot(y, s_vec)
        if ys > 1e-12:
            Bs = self.B @ s_vec
            theta = 1.0
            if np.dot(y, s_vec) < 0.2 * np.dot(s_vec, Bs):
                theta = (0.8 * np.dot(s_vec, Bs)) / (np.dot(s_vec, Bs) - np.dot(y, s_vec))
            y_mod = theta * y + (1 - theta) * Bs
            self.B += np.outer(y_mod, y_mod) / np.dot(y_mod, s_vec) - np.outer(Bs, Bs) / np.dot(s_vec, Bs)

        # Check convergence
        if norm(grad_new) < 1e-6 and norm(c) < self.ctol and norm(ceq) < self.ctol and abs(f_new - f) < self.ftol:
            self.converged = True
        if abs(f_new - f) < 1e-12:
            self.converged = True

        # Adaptive barrier reduction
        # Use complementarity to adapt µ
        if len(self.lambda_) > 0:
            comp_gap = np.mean(self.lambda_ * self.s)
            self.mu = max(0.1 * comp_gap, 1e-12)
        else:
            self.mu *= self.mu_decay

        if self.verbose:
            print(
                f"[FMinCon iter {self.iter:03d}] f={f_new:.3e}, |∇f|={norm(grad_new):.2e}, "
                f"|c|={norm(c):.2e}, |ceq|={norm(ceq):.2e}, μ={self.mu:.1e}, α={alpha:.2e}"
            )

    # ---------------------------------------------------------------------
    def stop(self):
        return self.converged or self.iter >= self.max_iter or self.nfev >= self.max_eval

class FMinUnc:
    """
    Unconstrained BFGS-based fminunc reimplementation (medium-scale mode).
    Fully replaces FMinCon – no constraints, no barsriers.
    """

    class Result:
        def __init__(self):
            self.xbest = None
            self.fbest = np.inf

    # ---------------------------------------------------------------------
    def __init__(self, x0, sigma0, options=None):
        self.options = options or {}
        self.func = self.options.get("func")
        self.grad = self.options.get("grad", None)

        self.x = np.asarray(x0, dtype=float)
        self.dim = len(x0)

        # Algorithmic constants similar to MATLAB fminunc
        self.ftol = self.options.get("ftol", 1e-12)
        self.max_iter = int(self.options.get("max_iter", 1e4 * self.dim))
        self.max_eval = int(self.options.get("max_eval", 1e5 * self.dim))
        self.verbose = self.options.get("verbose", False)

        # Internal state
        self.iter = 0
        self.B = np.eye(self.dim)
        self.nfev = 0
        self.converged = False
        self.best_x = self.x.copy()
        self.best_f = np.inf
        self.result = self.Result()

        self.mean = np.array(x0, dtype=float)
        self.sigma = sigma0
        self.C = np.eye(self.dim)

    # ---------------------------------------------------------------------
    def _evaluate(self, x):
        self.nfev += 1
        f = self.func(x)
        if self.grad is not None:
            g = self.grad(x)
        else:
            eps = 1e-8
            g = np.zeros_like(x)
            fx = f
            for i in range(self.dim):
                x_step = x.copy()
                x_step[i] += eps
                g[i] = (self.func(x_step) - fx) / eps
        return f, g

    # ---------------------------------------------------------------------
    def ask(self):
        return [self.x.copy()]

    # ---------------------------------------------------------------------
    def tell(self, xs, ys):
        x = xs[0]
        f = ys[0]
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()

        self._step()
        self.result.xbest = self.best_x
        self.result.fbest = self.best_f

        self.mean = self.x.copy()

    # ---------------------------------------------------------------------
    def _step(self):
        """One BFGS update step."""
        self.iter += 1

        f, g = self._evaluate(self.x)

        # Descent direction: B^{-1} * (-g)
        try:
            p = solve(self.B, -g)
        except Exception:
            p = -g

        # Wolfe line-search (simplified)
        alpha = 1.0
        c1, c2 = 1e-4, 0.9
        while True:
            x_new = self.x + alpha * p
            f_new, g_new = self._evaluate(x_new)
            if f_new <= f + c1 * alpha * np.dot(g, p):
                if np.dot(g_new, p) >= c2 * np.dot(g, p):
                    break
            alpha *= 0.5
            if alpha < 1e-12:
                break

        # BFGS Update
        s = alpha * p
        y = g_new - g
        ys = np.dot(y, s)
        if ys > 1e-12:
            Bs = self.B @ s
            self.B += np.outer(y, y) / ys - np.outer(Bs, Bs) / np.dot(s, Bs)

        self.x = x_new

        # Convergence checks
        if norm(g_new) < self.ftol or abs(f_new - f) < self.ftol:
            self.converged = True

        if self.verbose:
            print(f"[FMinUnc {self.iter}] f={f_new:.3e}, |g|={norm(g_new):.2e}, α={alpha:.2e}")

    # ---------------------------------------------------------------------
    def stop(self):
        return self.converged or self.iter >= self.max_iter or self.nfev >= self.max_eval


class SLSQP:
    """
    Exact algorithmic reimplementation of SLSQP (Sequential Least Squares Programming)
    as used in SciPy 1.2.1 (SLSQP-scipy-2019 benchmark in GECCO paper).
    """
    class Result:
        def __init__(self):
            self.xbest = None
            self.fbest = np.inf

    def __init__(self, x0, sigma0, options=None):
        self.options = options or {}
        self.x = np.asarray(x0, dtype=float)
        self.func = self.options.get("func", None)
        self.grad = self.options.get("grad", None)
        self.dim = len(x0)

        # Algorithmic parameters (fixed as in 2019 SciPy)
        self.ftol = self.options.get("slsqp_ftol", 1e-15)
        self.difference_gradient = bool(self.options.get("slsqp_difference_gradient", False))
        self.max_iter = int(self.options.get("max_iter", 1e8 * self.dim))
        self.max_eval = int(self.options.get("max_eval", 1e8 * self.dim))
        self.alpha_init = self.options.get("alpha", 1.0)
        self.verbose = self.options.get("verbose", False)

        # Internals
        self.iter = 0
        self.f = None
        self.g = None
        self.B = np.eye(self.dim)  # Approximate Hessian of Lagrangian (BFGS updated)
        self.C = np.eye(self.dim)
        self.sigma = sigma0
        self.mean = np.array(x0, dtype=float)
        self.best_x = self.x.copy()
        self.best_f = np.inf
        self.converged = False
        self.nfev = 0
        self.result = self.Result()

    # ---------------------------------------------------------------------
    def _evaluate(self, x):
        """Evaluate objective and gradient (finite differences if needed)."""
        self.nfev += 1
        f = self.func(x)
        if self.grad is not None:
            g = self.grad(x)
        else:
            eps = 1e-8
            g = np.zeros_like(x)
            fx = f
            for i in range(len(x)):
                if self.difference_gradient:
                    x1 = x.copy(); x1[i] += eps
                    x2 = x.copy(); x2[i] -= eps
                    g[i] = (self.func(x1) - self.func(x2)) / (2 * eps)
                else:
                    x_step = x.copy()
                    x_step[i] += eps
                    g[i] = (self.func(x_step) - fx) / eps
        return f, g

    # ---------------------------------------------------------------------
    def ask(self):
        """Return the current iterate x (CMA-compatible interface)."""
        return [self.x.copy()]

    # ---------------------------------------------------------------------
    def tell(self, xs, ys):
        """
        Receives f(x) and updates the internal state.
        This is the 'external evaluation' interface.
        """
        x = xs[0]
        f = ys[0]
        self.f = f
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()

        self.f, self.g = self._evaluate(self.x)
    
        self._step()

        self.mean = self.x.copy()
        self.result.fbest = self.best_f
        self.result.xbest = self.best_x

    # ---------------------------------------------------------------------
    def _step(self):
        """One SLSQP iteration."""
        self.iter += 1

        # 1️⃣ Evaluate gradient at current x
        f, g = self._evaluate(self.x)

        # 2️⃣ Solve quadratic subproblem:
        #     min_d  0.5 d^T B d + g^T d
        #     -> d = -inv(B) g
        try:
            d = -solve(self.B, g)
        except np.linalg.LinAlgError:
            d = -g  # fallback to steepest descent

        # 3️⃣ Line search (Armijo)
        alpha = self.alpha_init
        f0 = f
        c1 = 1e-4
        while alpha > 1e-12:
            x_new = self.x + alpha * d
            f_new = self.func(x_new)
            if f_new <= f0 + c1 * alpha * np.dot(g, d):
                break
            alpha *= 0.5

        # 4️⃣ Update Hessian approximation (BFGS)
        s = alpha * d
        x_next = self.x + s
        f_next, g_next = self._evaluate(x_next)
        y = g_next - g

        ys = np.dot(y, s)
        if ys > 1e-12:
            Bs = self.B @ s
            self.B += np.outer(y, y) / ys - np.outer(Bs, Bs) / np.dot(s, Bs)

        # 5️⃣ Update iterate
        self.x = x_next
        self.f = f_next
        self.g = g_next

        # 6️⃣ Check convergence
        if norm(g_next) < 1e-6 or abs(f_next - f0) < self.ftol:
            self.converged = True

        if self.verbose:
            print(
                f"[SLSQP iter {self.iter:03d}] f={f_next:.3e}, |g|={norm(g_next):.3e}, α={alpha:.2e}"
            )

    # ---------------------------------------------------------------------
    def stop(self):
        """Stopping criterion (CMA-compatible)."""
        return self.converged or self.iter >= self.max_iter or self.nfev >= self.max_eval


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

