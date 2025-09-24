import random

from sklearn.neighbors import NearestNeighbors
from scipy.stats import qmc, chi2, kendalltau
import numpy as np

from cma_es import CMAEvolutionStrategy
    

class Optimizer:
    def __init__(self):
        pass

    def initialize(self, args):
        """
        Initialize optimizer:
            - returns: X_obs, y_obs, es, surrogate_life, surrogate_user
        """
        # Initialize cs and X_obs, y_obs
        if args.init == "zeros":
            es = CMAEvolutionStrategy(args.dimension * [0], self.sigma, {'popsize': self.pop_size, 'cma_type': args.cma_type, 'CMA_active': args.active_cma})
            X_obs, y_obs = [], []
        elif args.init == "lhs":
            lhs = qmc.LatinHypercube(d=args.dimension)
            X_init = lhs.random(n=3*self.pop_size)
            y_init = [self.eval_problem(x) for x in X_init]

            # 3. Najdi nejlepší bod jako start CMA-ES
            x0 = X_init[np.argmin(y_init)]
            es = CMAEvolutionStrategy(x0, 1.0, {'popsize': self.pop_size, 'cma_type': args.cma_type, 'CMA_active': args.active_cma})
            X_obs, y_obs = list(X_init), list(y_init)

        # Initialize surrogate life
        surrogate_life = 1
        surrogate_used = 0

        return X_obs, y_obs, es, surrogate_life, surrogate_used


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
            ys = [self.eval_problem(x) for x in xs]
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
            args.model.update(data)

        if args.elitism:
            self.last_xs = xs if xs is not None else None
            self.last_y_pred_raw = y_pred_raw if y_pred_raw is not None else None
            self.last_y_pred = ys if ys is not None else None
        
        return top_idx, xs, ys, y_pred_raw, y_pred, xs_real_evaluated, ys_real_evaluated

    def eval_problem(self, x):
        if tuple(x) in self.eval_values.keys():
            return self.eval_values[tuple(x)]
        else:
            value = self.problem(x)
            self.eval_values[tuple(x)] = value
            self.evals += 1
            return value

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

    def tell(self, args, es, xs, ys, xs_real_evaluated, ys_real_evaluated):
        """
        Tell new evaluated points to cma.
        """
        if args.ask_by == "cma":
            if args.cma_only_real:
                es.tell(xs_real_evaluated, ys_real_evaluated)
            else:
                es.tell(xs, ys)

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
        if args.pop_increase[-5:] == "bipop":
            budget_small = 0
            budget_large = 0
            large_pop_size = self.pop_size
            large_sigma = self.sigma

        while self.evals < args.max_evals_per_dim * args.dimension:
            X_obs, y_obs, es, surrogate_life, surrogate_used = self.initialize(args)

            data = {"es": es}
            args.model.update(data)
            iteration = 0
            evals_at_start = self.evals
            self.last_xs, self.last_y_pred_raw, self.last_y_pred = None, None, None
            while self.evals < args.max_evals_per_dim * args.dimension:
                # Ask for new X values
                xs = self.ask(args, X_obs, y_obs, es)

                # If enough data, use surrogate for pre-evaluation
                if (not args.model.is_empty()) and len(X_obs) >= self.pop_size:
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
                        _, xs, ys, _, _, xs_real_evaluated, ys_real_evaluated = self.evaluate_real(args, es, xs, None, None, evaluate_all = True)

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
                        top_idx, xs, ys, y_pred_raw, y_pred, xs_real_evaluated, ys_real_evaluated = self.evaluate_real(args, es, xs, y_pred, y_pred_raw)

                        # Update surrogate lifelength
                        surrogate_life, surrogate_used = self.update_surrogate_life(args, surrogate_life, surrogate_used, top_idx, y_pred_raw, ys_real_evaluated)
                        
                else:
                    _, xs, ys, _, _, xs_real_evaluated, ys_real_evaluated = self.evaluate_real(args, es, xs, None, None, evaluate_all = True)

                # Update CMA
                self.tell(args, es, xs, ys, xs_real_evaluated, ys_real_evaluated)

                # Extend observations
                X_obs, y_obs = self.extend_observations(args, X_obs, y_obs, xs, ys, xs_real_evaluated, ys_real_evaluated)
                
                print(f"[{args.experiment_name} Iter {iteration} Evals/dim {self.evals//args.dimension}] f(best)={np.min(ys_real_evaluated if len(ys_real_evaluated)>0 else ys):.5f}")
                if args.ask_by == "cma" and es.stop():
                    break
                iteration += 1
                
            if args.pop_increase == "none":
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
                    
        print(f"✅ {args.experiment_name} Done. Best:", es.result.xbest)

        
