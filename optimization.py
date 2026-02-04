import random
import copy

from sklearn.neighbors import NearestNeighbors
from scipy.stats import qmc, chi2, kendalltau
import numpy as np
from datetime import datetime
from scipy.optimize import fmin_slsqp, minimize

from cma_es import CMAEvolutionStrategy, StepND

class COCORandom:
    """COCO-style deterministic random number generator."""
    def __init__(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def gauss(self, mu=0.0, sigma=1.0):
        return self.rng.normal(mu, sigma)

def compute_coco_fopt(function_id: int, instance_id: int) -> float:
    """
    Compute deterministic global optimum value (fopt) as defined by COCO/BBOB.
    Returns a float value for given function and instance IDs.
    """
    seed = function_id + 10000 * instance_id
    rng = COCORandom(seed)
    fopt = min(1000.0, max(-1000.0, rng.gauss() * 100.0 / rng.gauss()))
    return fopt

def get_fopt_from_problem(problem) -> float:
    """
    Extract function_id and instance_id from a cocoex Problem object
    and compute its fopt (global optimum).
    """
    f_id = problem.id_function
    i_id = problem.id_instance
    return compute_coco_fopt(f_id, i_id)
    

class Optimizer:
    def __init__(self):
        pass

    def initialize(self, args):
        """
        Initialize optimizer:
            - returns: X_obs, y_obs, es, surrogate_life, surrogate_user
        """
        self.args = args
        # Initialize cs and X_obs, y_obs
        x0 = args.dimension * [0]
        if hasattr(args, "x0"):
            x0 = args.x0
        elif args.init == "lhs":
            lhs = qmc.LatinHypercube(d=args.dimension)
            X_init = lhs.random(n=3*self.pop_size)
            y_init = [self.eval_problem(x) for x in X_init]
            x0 = X_init[np.argmin(y_init)]
        elif args.init == "slsqp":
            x0 = fmin_slsqp(
                func=self.eval_problem,
                x0=np.array(x0, dtype=float),
                iter=args.init_slsqp_iterations,
                acc=args.init_slsqp_accuracy,
                iprint=0,
                full_output=False,
            )
        elif args.init == "sobol":
            sobol = qmc.Sobol(d=args.dimension, scramble=True)
            # Generate the same number of initial points as with LHS
            X_init = sobol.random(n=3*self.pop_size)
            # Ensure domain is respected if bounds exist (default assumes unit cube)
            if hasattr(self, "lb") and hasattr(self, "ub"):
                X_init = qmc.scale(X_init, self.lb, self.ub)

            y_init = [self.eval_problem(x) for x in X_init]
            x0 = X_init[np.argmin(y_init)]
                    
        es = CMAEvolutionStrategy(x0, self.sigma, {
            'popsize': self.pop_size, 'cma_type': args.cma_type, 'CMA_active': args.active_cma,
            'func': self.eval_problem, "optimizer": args.optimizer, 'slsqp_ftol': args.slsqp_ftol,
            'slsqp_difference_gradient': args.slsqp_difference_gradient, 'hees_tol': args.hees_tol,
            'shade_lm': args.shade_lm
            })
        
        X_obs, y_obs = [], []
        if args.init == "lhs":
            X_obs, y_obs = list(X_init), list(y_init)
        if args.local_search_check:
            self.solutions = []

        # Initialize surrogate life
        surrogate_life = 1
        surrogate_used = 0

        return X_obs, y_obs, es, surrogate_life, surrogate_used, x0


    def ask(self, args, X_obs, y_obs, es):
        """
        Asks for new X data
        """
        xs = None
        if args.ask_by == "cma":
            xs = np.asarray(es.ask())
            if args.center_injection:
                xs = np.concatenate([xs, [es.mean.copy()]])
        elif args.ask_by == "around_best":
            if len(X_obs) > 0:
                center = X_obs[np.argmin(y_obs)]
                xs = center + np.random.normal(0, 0.3, size=(self.pop_size, args.dimension))
            else:
                xs = np.random.uniform(-5, 5, size=(self.pop_size, args.dimension))
        return xs

    def choose_train_data(self, args, X_obs, y_obs, xs, es):
        """
        Returns train data from all observations X_obs, y_obs
        """
        if args.tss == "full":
            idx_raw = np.argsort(y_obs)
        elif args.tss == "knn":
            m_dim = (args.dimension * (args.dimension + 3)) // 2 + 1
            k = int(np.ceil(min(2 * m_dim, np.sqrt(len(X_obs) * m_dim))))
            
            nn = NearestNeighbors(n_neighbors=min(k, len(X_obs)), algorithm='auto')
            nn.fit(X_obs)

            # For each query, get k nearest neighbors in archive
            all_indices = []
            for x in xs:
                _, idx = nn.kneighbors([x])
                all_indices.extend(idx[0])

            # Union of all indices
            idx_raw = np.unique(all_indices)
        elif args.tss == "nearest":
            r_max = 4 * np.sqrt(chi2.ppf(0.99, args.dimension))
            N_max = 20 * args.dimension
            
            selected = set()
            k = 1
            while True:
                new_selected = set()
                nn = NearestNeighbors(n_neighbors=min(k, len(X_obs)))
                nn.fit(X_obs)

                # Union of k-NN sets for all candidates
                for x in xs:
                    dists, idx = nn.kneighbors([x], return_distance=True)
                    for j, d in zip(idx[0], dists[0]):
                        if np.linalg.norm(X_obs[j] - es.mean) <= r_max:
                            new_selected.add(j)

                # Check stopping conditions
                if (len(new_selected) > N_max) or (len(new_selected) == len(selected)):
                    break
                selected = new_selected
                k += 1
            idx_raw = list(selected)
                
        X_raw = np.array([X_obs[i] for i in idx_raw])
        y_raw = np.array([y_obs[i] for i in idx_raw])

        return X_raw, y_raw

    def scale_data(self, args, X_raw, y_raw, xs, es):
        """
        Returns scaled data for X and y
        """
        # Scale (preprocess) data
        if args.scaler:
            data = {'X': np.array(X_raw), 'es': es}
            X_scaled_obs = args.scaler.fit_transform(data)
            X_scaled = args.scaler.transform(xs)
        else:
            X_scaled_obs = np.asarray(X_raw)
            X_scaled = np.asarray(xs)

        # Scale (preprocess) y data
        if args.y_scaler:
            data = {'X': np.array(y_raw)}
            y_scaled_obs = args.y_scaler.fit_transform(data)
        else:
            y_scaled_obs = np.asarray(y_raw)

        return X_scaled_obs, X_scaled, y_scaled_obs

    def fit_model(self, args, X_scaled_obs, y_scaled_obs, surrogate_life, surrogate_used):
        """
        Fits model, also updates surrogate_used.
        """
        if (not args.check_surrogate_life) or surrogate_used >= surrogate_life:
            if args.use_pair_model:
                X_pairs, y_pairs = [], []
                combined = list(zip(X_scaled_obs, y_scaled_obs))
                random.shuffle(combined)
                X_for_pairs, y_for_pairs = zip(*combined)
                X_for_pairs, y_for_pairs = list(X_for_pairs), list(y_for_pairs)
                for i in range(len(X_for_pairs)):
                    for j in range(i + 1, len(X_for_pairs)):
                        xi, xj = X_for_pairs[i], X_for_pairs[j]
                        yi, yj = y_for_pairs[i], y_for_pairs[j]
                        if yi == yj:
                            continue
                        X_pairs.append(xi - xj)
                        y_pairs.append(1 if yi < yj else 0)
                args.model.fit(X_pairs, y_pairs)                            
            else:
                args.model.fit(X_scaled_obs, y_scaled_obs)
            surrogate_used = 0

        return surrogate_used

    def predict_model(self, args, X_scaled, X_scaled_obs):
        """
        Predict model to get y predictions.
        """
        y_pred_raw, sigma = None, None
        if args.prediction_transform == "ucb" or args.prediction_transform == "ilcb":
            y_pred_raw, sigma = args.model.predict(X_scaled, return_std=True)
        elif args.use_pair_model:
            scores = []
            for x in X_scaled:
                diffs = [x - xi for xi in X_scaled_obs]
                preds = args.model.decision_function(diffs)
                scores.append(np.mean(preds))
            y_pred_raw = np.asarray(scores)
        else:
            y_pred_raw = args.model.predict(X_scaled)

        return y_pred_raw, sigma

    def is_flat(self, args, y_scaled_obs, y_pred_raw):
        """
        Returns if y_pred_raw is flat in comparison to y_scaled_obs.
        """
        return (np.max(y_pred_raw) - np.min(y_pred_raw)) < min(1e-8, args.flatness_alpha * (np.max(y_scaled_obs) - np.min(y_scaled_obs)))

    def differential_evolution_block(self, f, X, F=0.5, CR=0.5, generations=4, bounds=None):
        """
        Perform a few DE generations on population X.
        Exponential crossover and best/2 mutation as described in Pal (2016).
        """
        pop = X.copy()
        n, d = pop.shape
        fvals = np.array([f(ind) for ind in pop])

        for _ in range(generations * d):
            best = pop[np.argmin(fvals)]
            new_pop = []
            for i in range(n):
                idxs = np.random.choice([j for j in range(n) if j != i], 4, replace=False)
                r1, r2, r3, r4 = pop[idxs]

                # Mutation: best + F*(r1 - r2) + F*(r3 - r4)
                mutant = best + F * (r1 - r2) + F * (r3 - r4)

                # Exponential crossover
                start = np.random.randint(d)
                L = 0
                while np.random.rand() < CR and L < d:
                    L += 1
                end = (start + L) % d
                trial = pop[i].copy()
                if end > start:
                    trial[start:end] = mutant[start:end]
                else:
                    trial[start:] = mutant[start:]
                    trial[:end] = mutant[:end]

                # Bound handling (if any)
                if bounds is not None:
                    trial = np.clip(trial, bounds[:, 0], bounds[:, 1])

                f_trial = f(trial)
                if f_trial < fvals[i]:
                    pop[i], fvals[i] = trial, f_trial

            # Replace population
            pop = new_pop if len(new_pop) else pop

        return pop, fvals

    def _distance_filter(self, args, x):
        """Check if x is far enough from known local minima."""
        if not self.solutions:
            return True
        for x_star, _ in self.solutions:
            if np.linalg.norm(x - x_star) < args.local_search_distance_tol:
                return False
        return True

    def _merit_filter(self, args, fval):
        """Check if x has merit better than previous local minima."""
        if not self.solutions:
            return True
        best_f = min(f for _, f in self.solutions)
        return fval < best_f - args.local_search_merit_tol

    def evaluate_real(self, args, es, xs, y_pred, y_pred_raw, evaluate_all = False):
        """
        Evaluate X by real problem
        """
        orig_xs_shape = np.asarray(xs).shape
        if args.elitism:
            if evaluate_all or (self.last_y_pred_raw is not None and self.last_xs is not None and self.last_y_pred is not None and len(self.last_xs) == len(self.last_y_pred_raw) and len(self.last_xs) == len(self.last_y_pred)):
                xs = np.vstack([self.last_xs, xs]) if self.last_xs is not None else np.array(xs)
                if not evaluate_all:
                    y_pred_raw = np.concatenate([self.last_y_pred_raw, y_pred_raw]) if self.last_y_pred_raw is not None else y_pred_raw
                    y_pred = np.concatenate([self.last_y_pred, y_pred]) if self.last_y_pred is not None else y_pred
            
        top_idx = None
        if evaluate_all:
            if self.args.eval_problem == "standard":
                ys = [self.eval_problem(x) for x in xs]
            elif self.args.eval_problem == "local_search":
                ys = [self.eval_problem(x) for x in xs]
                xs_idx = np.argsort(ys)[:orig_xs_shape[0]]
                
                xs_reduced = np.asarray(xs)[xs_idx][:int(args.mlsl_gamma * len(xs))]
                ys_reduced = np.asarray(ys)[xs_idx][:int(args.mlsl_gamma * len(xs))]

                if args.local_search_de_refinement and len(xs_reduced) > 6:
                    xs_reduced, _ = self.differential_evolution_block(self.eval_problem, xs_reduced)
                
                for i in range(len(xs_reduced)):
                    x = xs_reduced[i]
                    y = ys_reduced[i]
                    if (not args.local_search_check) or (self._distance_filter(args, x) and self._merit_filter(args, y)):
                        local_budget = int(self.args.local_search_frac * (self.args.max_evals_per_dim * self.args.dimension) / self.args.pop_size)
                        res = minimize(
                            self.eval_problem,
                            x,
                            method="trust-constr",
                            tol=1e-12,
                            options={
                                "maxiter": local_budget,
                                "verbose": 0,
                                "gtol": 1e-8,
                                "xtol": 1e-12,
                                "barrier_tol": 1e-12,
                            },
                        )
                        if args.local_search_check:
                            self.solutions.append((res.x, res.fun))
            xs_idx = np.argsort(ys)[:orig_xs_shape[0]]
            xs, ys = np.asarray(xs)[xs_idx], np.asarray(ys)[xs_idx]
            y_pred_raw = ys
            y_pred = ys
            xs_real_evaluated, ys_real_evaluated = xs, ys
        else:
            xs_idx = np.argsort(y_pred)[:orig_xs_shape[0]]            
            xs, y_pred_raw, y_pred = np.asarray(xs)[xs_idx], np.asarray(y_pred_raw)[xs_idx], np.asarray(y_pred)[xs_idx]
            
            top_idx = np.argsort(y_pred)[:int(len(xs)*args.real_evaluation_ratio.get())]

            # Generate new ys
            ys = [self.eval_problem(xs[i]) if i in top_idx else y_pred[i] for i in range(len(xs))]
                    
            xs_real_evaluated = [xs[i] for i in top_idx]
            ys_real_evaluated = [ys[i] for i in top_idx]

            data = {"pred": [y_pred_raw[i] for i in top_idx], "true": ys_real_evaluated, "xs": xs_real_evaluated, "es": es}
            args.real_evaluation_ratio.update(data)

        if args.elitism:
            self.last_xs = xs if xs is not None else None
            self.last_y_pred_raw = y_pred_raw if y_pred_raw is not None else None
            self.last_y_pred = ys if ys is not None else None

        xs_inv_idx = np.argsort(xs_idx)
        
        return top_idx, xs_inv_idx, xs, ys, y_pred_raw, y_pred, xs_real_evaluated, ys_real_evaluated

    def eval_problem(self, x):
        if tuple(x) in self.eval_values.keys():
            return self.eval_values[tuple(x)]
        else:
            value = self.problem(x)
            self.eval_values[tuple(x)] = value
            self.evals += 1
            return value
        
            return self._eval_problem(x)
        
            
            return self._eval_problem(x)

    def inverse_scale_data(self, args, y_pred_raw):
        """
        Inverse scale y predictions
        """
        if args.y_scaler:
            y_pred_raw = args.y_scaler.inverse_transform(y_pred_raw)
        return y_pred_raw

    def transform_predictions(self, args, y_pred_raw, sigma, X_raw, X_scaled_obs, y_scaled_obs):
        """
        Transform predicion - UCB/ILCB/None
        """
        if args.prediction_transform == "ucb":
            beta_t = 2 * np.log(len(X_raw) + 1)
            y_pred = y_pred_raw - np.sqrt(beta_t) * sigma
        elif  args.prediction_transform == "ilcb":
            mu_obs, std_obs = args.model.predict(np.asarray(X_scaled_obs), return_std=True)
            std_obs = np.maximum(std_obs, 1e-9)
            z = np.abs((np.asarray(y_scaled_obs) - mu_obs) / std_obs)              # standardized residuals
            ilcb_kappa = float(np.clip(np.quantile(z, 0.90), 0.5, 5.0))
            y_pred = y_pred_raw - ilcb_kappa * sigma
        else:
            y_pred = y_pred_raw
        return y_pred

    def update_surrogate_life(self, args, surrogate_life, surrogate_used, top_idx, y_pred_raw, ys_real_evaluated):
        """
        Updates surrogate life
        """
        if args.check_surrogate_life:
            preds_real_evaluated = [y_pred_raw[i] for i in top_idx]
            error = np.mean(np.abs(np.asarray(ys_real_evaluated) - np.asarray(preds_real_evaluated)))
                        
            if error < 1e-2:   # good surrogate
                surrogate_life = min(surrogate_life + 1, 10)
            else:              # bad surrogate
                surrogate_life = max(surrogate_life // 2, 1)

            surrogate_used += 1
        return surrogate_life, surrogate_used

    def tell(self, args, es, xs, ys, xs_real_evaluated, ys_real_evaluated, xs_inv_idx, xs_len):
        """
        Tell new evaluated points to cma.
        """
        if args.ask_by == "cma":
            if args.cma_only_real and not args.optimizer == "hees":
                es.tell(xs_real_evaluated, ys_real_evaluated)
            else:
                es.tell(xs, ys, xs_inv_idx, xs_len)

    def extend_observations(self, args, X_obs, y_obs, xs, ys, xs_real_evaluated, ys_real_evaluated):
        if args.train_only_real:
            X_obs.extend(xs_real_evaluated)
            y_obs.extend(ys_real_evaluated)
        else:
            X_obs.extend(xs)
            y_obs.extend(ys)
        return X_obs, y_obs

    def get_kendalltau(self, args, X_scaled_obs, y_raw):
        y_pred_raw, sigma = self.predict_model(args, X_scaled_obs, X_scaled_obs)
        y_pred_raw = self.inverse_scale_data(args, y_pred_raw)
        tau, _ = kendalltau(y_raw, y_pred_raw)
        #print(f"tau: {tau}")
        return tau

    def chaotic_local_search(self, x, sigma, R_init=0.3, n_points=50):
        dim = len(x)
        r = np.random.rand(dim)
        chaos_seq = []
        for _ in range(n_points):
            r = np.where(r < 0.7, r / 0.7, 10 * (1 - r) / 3)  # tent map
            chaos_seq.append(r.copy())
        chaos_seq = np.array(chaos_seq)
        R = sigma * R_init
        return x + R * (2 * chaos_seq - 1)
        
        
    def optimize(self, problem, args):
        """
            args.scaler - scaler (prerocessor)
            args.model - surrogate model
            args.real_evaluation_ratio - ratio of real evalutated data
            args.cma_only_real - if only real evaluation is used to cma
            args.train_only_real - if only real evaluation is used as training data
            args.experiment_name - name of experiment
        """
        # Initialization
        self.problem = problem
        self.evals = 0
        self.eval_values = {}
        self.pop_size = args.pop_size
        self.sigma = args.sigma
        self.rng = np.random.RandomState()
        self.f_best = 99999999999
        if args.pop_increase[-5:] == "bipop":
            budget_small = 0
            budget_large = 0
            large_pop_size = self.pop_size
            large_sigma = self.sigma

        while self.evals < args.max_evals_per_dim * args.dimension:
            X_obs, y_obs, es, surrogate_life, surrogate_used, x0 = self.initialize(args)

            if args.step:
                step_alg = StepND(self.eval_problem, copy.deepcopy(x0), step_size=self.sigma, tol=args.step_tol)
                evals_total = 0
                evals_cma = 0
                evals_step = 0
                active_step = True
                best_cma = self.eval_problem(x0)
                best_step = best_cma
                best_overall = best_cma

            data = {"es": es}
            args.model.update(data)
            iteration = 0
            evals_at_start = self.evals
            self.last_xs, self.last_y_pred_raw, self.last_y_pred = None, None, None
            while self.evals < args.max_evals_per_dim * args.dimension:
                evals_at_iter_start = self.evals
                # Ask for new X values
                if args.step and active_step and evals_step / max(1, evals_total) < args.step_rho:
                    xs = step_alg.ask()
                else:
                    xs = self.ask(args, X_obs, y_obs, es)
                xs_len = len(xs)

                # If enough data, use surrogate for pre-evaluation
                if (not args.model.is_empty()) and len(X_obs) >= self.pop_size and len(X_obs) >= 40:
                    # Fit and predict model
                    fitted = False
                    last_ranking = np.zeros(len(xs), dtype = np.int16)
                    last_set = np.zeros(len(xs)//2, dtype = np.int16)
                    last_best = -1
                    evaluated = []
                    while (not fitted) or args.fit_until in ["unchanged_ranking", "unchanged_set"]:
                        # Choose train data
                        X_raw, y_raw = self.choose_train_data(args, X_obs, y_obs, xs, es)

                        # Scale data (X, y)
                        X_scaled_obs, X_scaled, y_scaled_obs = self.scale_data(args, X_raw, y_raw, xs, es)
                        
                        # Fit model
                        surrogate_used = self.fit_model(args, X_scaled_obs, y_scaled_obs, surrogate_life, surrogate_used)
                        fitted = True

                        # Predict model
                        y_pred_raw, sigma = self.predict_model(args, X_scaled, X_scaled_obs)

                        if args.fit_until in ["unchanged_ranking", "unchanged_set"]:
                            if len(y_pred_raw.shape) > 1:
                                y_pred_raw = y_pred_raw[:,0]
                            ranking = np.argsort(y_pred_raw)
                            new_set = set(np.argsort(y_pred_raw)[: len(xs) // 2])
                            best = int(np.argmin(y_pred_raw))
                            if args.fit_until == "unchanged_ranking" and np.array_equal(ranking, last_ranking):
                                break
                            elif args.fit_until == "unchanged_set" and ((len(evaluated) >= len(xs)//4 and best == last_best) or (best == last_best and new_set == last_set)):
                                break
                            else:
                                last_ranking = ranking
                                last_set = new_set
                                last_best = best
                                nb = max(1, len(xs) // 10)
                                to_eval = ranking[len(evaluated): len(evaluated)+nb]
                                if not len(to_eval): break
                                xs_real = [xs[i] for i in to_eval]
                                ys_real = [self.eval_problem(xs[i]) for i in to_eval]
                                evaluated.extend(to_eval)
                                X_obs, y_obs = self.extend_observations(args, X_obs, y_obs, [], [], xs_real, ys_real)
                        

                    # Check if predictions are not flat
                    if (args.check_flatness and self.is_flat(args, y_scaled_obs, y_pred_raw)) or (args.check_train_kendalltau and self.get_kendalltau(args, X_scaled_obs, y_raw) <= 0.85):
                        print("⚠️ Surrogate model marked as constant (not used)")
                        _, xs_inv_idx, xs, ys, _, _, xs_real_evaluated, ys_real_evaluated = self.evaluate_real(args, es, xs, None, None, evaluate_all = True)

                    else:
                        # Inverse scale predictions
                        y_pred_raw = self.inverse_scale_data(args, y_pred_raw)

                        # Transform predictions
                        y_pred = self.transform_predictions(args, y_pred_raw, sigma, X_raw, X_scaled_obs, y_scaled_obs)

                        if args.chaotic_extend:
                            screened_size = self.pop_size//2
                            compet_size = self.pop_size//2
                            idx = np.argsort(y_pred)[:screened_size]
                            X_screened = [xs[i] for i in idx]
                            X_compet = []
                            y_pred_raw_compet = []
                            y_pred_compet = []
                            for i in idx[:compet_size]:
                                chaos = self.chaotic_local_search(xs[i], es.sigma)
                                _, chaos_scaled, _ = self.scale_data(args, X_raw, y_raw, chaos, es)
                                chaos_pred_raw, chaos_sigma = self.predict_model(args, chaos_scaled, X_scaled_obs)
                                chaos_pred_raw = self.inverse_scale_data(args, chaos_pred_raw)
                                chaos_pred = self.transform_predictions(args, chaos_pred_raw, chaos_sigma, X_raw, X_scaled_obs, y_scaled_obs)
                                X_compet.append(chaos[np.argmin(chaos_pred)])
                                y_pred_raw_compet.append(chaos_pred_raw[np.argmin(chaos_pred)])
                                y_pred_compet.append(chaos_pred[np.argmin(chaos_pred)])
                            xs = np.vstack([X_screened, X_compet])
                            y_pred = np.vstack([y_pred[:screened_size], y_pred_compet])
                            y_pred_raw = np.vstack([y_pred_raw[:screened_size], y_pred_raw_compet])
                            
                        # Evaluate best point by real problem
                        top_idx, xs_inv_idx, xs, ys, y_pred_raw, y_pred, xs_real_evaluated, ys_real_evaluated = self.evaluate_real(args, es, xs, y_pred, y_pred_raw)

                        # Update surrogate lifelength
                        surrogate_life, surrogate_used = self.update_surrogate_life(args, surrogate_life, surrogate_used, top_idx, y_pred_raw, ys_real_evaluated)
                        
                else:
                    _, xs_inv_idx, xs, ys, _, _, xs_real_evaluated, ys_real_evaluated = self.evaluate_real(args, es, xs, None, None, evaluate_all = True)

                data = {"true": ys_real_evaluated, "xs": xs_real_evaluated, "es": es}
                args.model.update(data)

                # Update CMA
                if args.step and active_step and evals_step / max(1, evals_total) < args.step_rho:
                    _, f_new, improved = step_alg.tell(xs, ys)
                    evals_step += self.evals - evals_at_iter_start
                    evals_total += self.evals - evals_at_iter_start
                    if improved and f_new < best_step:
                        best_step = f_new
                    if step_alg.iters > args.step_min_iters and step_alg.lines[0].points and step_alg.lines[0].f_star:
                        if step_alg.best_f > best_cma or not improved:
                            if step_alg.lines[0].f_star - step_alg.best_f < args.step_tol:
                                active_step = False
                else:
                    self.tell(args, es, xs, ys, xs_real_evaluated, ys_real_evaluated, xs_inv_idx, xs_len)
                    if args.step:
                        evals_cma += self.evals - evals_at_iter_start
                        evals_total += self.evals - evals_at_iter_start
                        fbest_cma = np.min(ys)
                        if fbest_cma < best_cma:
                            best_cma = fbest_cma

                if args.step and active_step and step_alg.lines[0].f_star > best_cma and step_alg.lines[0].f_star is not None:
                    if step_alg.lines[0].f_star - best_cma > args.step_tol and step_alg.lines[0].points:
                        active_step = False

                # Extend observations
                X_obs, y_obs = self.extend_observations(args, X_obs, y_obs, xs, ys, xs_real_evaluated, ys_real_evaluated)
                
                print(f"[{args.experiment_name} Iter {iteration} Evals/dim {self.evals//args.dimension}] f(best)={np.min(ys_real_evaluated if len(ys_real_evaluated)>0 else ys):.5f}")

                f_best = np.min(ys_real_evaluated if len(ys_real_evaluated)>0 else ys)
                if f_best < self.f_best:
                    self.f_best = f_best

                if problem.final_target_hit:
                    break
            
                if args.ask_by == "cma" and es.stop():
                    break
                iteration += 1
                
            if args.pop_increase == "none" or problem.final_target_hit:
                break
            elif args.pop_increase == "ipop":
                self.pop_size *= 2
            elif args.pop_increase == "nipop":
                self.pop_size *= 2
                self.sigma /= 1.6
            elif args.pop_increase[-5:] == "bipop":
                if budget_small < budget_large:
                    u = np.random.rand()
                    self.pop_size = int(args.pop_size * ((0.5 * large_pop_size / args.pop_size) ** (u ** 2)))
                    self.sigma = args.sigma * 10 ** (-2 * self.rng.rand())
                    budget_small += self.evals - evals_at_start
                else:
                    self.pop_size = 2 * large_pop_size
                    large_pop_size = self.pop_size
                    if args.pop_increase[0] == "n":
                        self.sigma = large_sigma / 1.6
                        large_sigma = self.sigma
                    else:
                        self.sigma = args.sigma
                    budget_large += self.evals - evals_at_start
                    
        args.model.save()

        print(f"[{datetime.now()}] Solved: {problem.final_target_hit}")
        print(f"    - Evaluations/dim: {self.evals/args.dimension:.4f}, BestF: {self.f_best}, Best: {es.result.xbest}")
        with open(f"results/{args.experiment_name}-{args.function_id}.txt", "a") as f:
            f.write(f"[{datetime.now()}] Solved: {problem.final_target_hit}\n")
            f.write(f"    - Evaluations/dim: {self.evals/args.dimension:.4f}, BestF: {self.f_best}, Best: {es.result.xbest}\n")

        return problem.final_target_hit, self.evals/args.dimension
      
