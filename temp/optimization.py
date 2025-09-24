
import numpy as np
import random
import cma
from typing import Callable

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel as C
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.exceptions import NotFittedError
from sklearn.base import clone
from scipy.interpolate import Rbf
from scipy.stats import qmc, norm
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F
from torch.optim import Adam
import torch
from collections import deque
import openai  # or any LLaMA3 API client
from numpy.linalg import inv

# Optional for transformer surrogate
from transformers import AutoModel, AutoTokenizer


# ----------------------------
# Hamiltonian Neural Network
# ----------------------------
class HamiltonianNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Predict Hamiltonian energy
        return self.fc(x)

    def grad(self, x):
        x.requires_grad_(True)
        H = self.forward(x)
        grad = torch.autograd.grad(H.sum(), x, create_graph=False)[0]
        return grad.detach()


# ----------------------------
# DQN Agent
# ----------------------------
class DQNAgent:
    def __init__(self, state_dim=4, action_dim=3, lr=1e-3, gamma=0.95, eps=0.1):
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.target_net = nn.Sequential(*[layer for layer in self.q_net])
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.eps = eps
        self.memory = deque(maxlen=5000)
        self.batch_size = 32

    def select_action(self, state):
        if random.random() < self.eps:
            return random.randint(0, 2)
        with torch.no_grad():
            return self.q_net(torch.FloatTensor(state)).argmax().item()

    def store(self, transition):
        self.memory.append(transition)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        q_vals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_vals = self.target_net(next_states).max(1)[0]
        expected_q = rewards + self.gamma * next_q_vals

        loss = nn.MSELoss()(q_vals, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ---------------- PPO Actor-Critic ----------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def act(self, state):
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)
        return action.item(), log_prob.detach(), value.detach()

    def evaluate(self, states, actions):
        probs = self.actor(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states).squeeze(-1)
        return log_probs, values, entropy

# ---------------- Rollout Buffer ----------------
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def store(self, state, action, log_prob, reward, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)

    def clear(self):
        self.states, self.actions, self.log_probs, self.rewards, self.values = [], [], [], [], []


class SolverConfig:
    def __init__(self, mutation_rate=None, step_size=None, pop_size=None):
        self.mutation_rate = mutation_rate or random.uniform(0.05, 0.5)
        self.step_size = step_size or random.uniform(0.1, 1.0)
        self.pop_size = pop_size or random.choice([10, 20, 30, 50])

    def mutate(self):
        # Mutate parameters
        if random.random() < 0.3:
            self.mutation_rate = max(0.01, min(0.9, self.mutation_rate + np.random.normal(0, 0.05)))
        if random.random() < 0.3:
            self.step_size = max(0.01, min(2.0, self.step_size + np.random.normal(0, 0.1)))
        if random.random() < 0.3:
            self.pop_size = max(5, min(100, self.pop_size + random.choice([-5, 5])))

    def crossover(self, other):
        # Simple uniform crossover
        child = SolverConfig(
            mutation_rate=random.choice([self.mutation_rate, other.mutation_rate]),
            step_size=random.choice([self.step_size, other.step_size]),
            pop_size=random.choice([self.pop_size, other.pop_size])
        )
        return child

# --- RBF surrogate (cubic, as in CMA-SAO Sec. 4.4) ---
class RBFModel:
    def __init__(self):
        self.X = None
        self.y = None
        self.weights = None
        self.centers = None

    def fit(self, X, y):
        """Fit cubic RBF: phi(r) = r^3"""
        self.X = np.array(X)
        self.y = np.array(y)
        self.centers = self.X.copy()

        # Build kernel matrix
        dists = cdist(self.X, self.centers)
        K = dists ** 3  # cubic RBF
        # Solve for weights (ridge regularization optional)
        self.weights = np.linalg.lstsq(K, self.y, rcond=None)[0]

    def predict(self, X):
        """Predict using cubic RBF"""
        if self.weights is None:
            raise RuntimeError("RBF not fitted")
        dists = cdist(np.atleast_2d(X), self.centers)
        K = dists ** 3
        return K.dot(self.weights)


# ---------------- Transformer surrogate ----------------
class TransformerSurrogate(nn.Module):
    def __init__(self, dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        h = self.transformer(x).squeeze(1)
        return self.head(h).squeeze(-1)
    

class Optimization:
    methods = {}
        
    # ----------------------------
    # Gaussian Process Initialization
    # ----------------------------
    @staticmethod
    def init_gp():
        kernel = C(1.0, (1e-3, 1e8)) * (
            Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e4), nu=2.5)
            + RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e4))
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=8
        )

    # ----------------------------
    # Utility: Select top-k candidates via UCB
    # ----------------------------
    @staticmethod
    def select_ucb_candidates(gp, xs, scaler, k=5):
        xs_scaled = scaler.transform(xs)
        mu, sigma = gp.predict(xs_scaled, return_std=True)
        ucb = mu - 1.96 * sigma
        return np.argsort(ucb)[:k]

    #def optimize()

    # ----------------------------
    # 1) Surrogate-assisted CMA-ES (S-CMA-ES)
    # ----------------------------
    @staticmethod
    def optimize_s_cma_es(problem, dim, max_iter, pop_size):
        es = cma.CMAEvolutionStrategy(dim * [0], 1.0, {'popsize': pop_size})
        gp = Optimization.init_gp()
        scaler = StandardScaler()
        X_obs, y_obs = [], []

        for iteration in range(max_iter):
            xs = es.ask()

            # If enough data, use GP surrogate for pre-evaluation
            if len(X_obs) >= pop_size:
                X_scaled_obs = scaler.fit_transform(np.array(X_obs))
                gp.fit(X_scaled_obs, np.array(y_obs))
                X_scaled = scaler.transform(xs)
                mu, sigma = gp.predict(X_scaled, return_std=True)
                # Choose top 50% for real evaluation
                top_idx = np.argsort(mu)[:len(xs)//2]
                ys = [problem(xs[i]) if i in top_idx else mu[i] for i in range(len(xs))]
            else:
                ys = [problem(x) for x in xs]

            X_obs.extend(xs)
            y_obs.extend(ys)
            es.tell(xs, ys)

            print(f"[S-CMA-ES Iter {iteration}] f(best)={np.min(ys):.5f}")
            if es.stop():
                break
        print("✅ S-CMA-ES Done. Best:", es.result.xbest)

    # ----------------------------
    # 2) DTS-CMA-ES (Double-Trust-Region)
    # ----------------------------
    @staticmethod
    def optimize_dts_cma_es(problem, dim, max_iter, pop_size):
        es = cma.CMAEvolutionStrategy(dim * [0], 1.0, {'popsize': pop_size})
        gp = Optimization.init_gp()
        scaler = StandardScaler()
        archive_X, archive_y = [], []

        for iteration in range(max_iter):
            # Step 1: Select archive points using simple TSS (most recent `pop_size * 3`)
            if len(archive_X) >= pop_size:
                A_idx = np.argsort(archive_y)[-pop_size * 3:]
                Xtr_raw = np.array([archive_X[i] for i in A_idx])
                ytr_raw = np.array([archive_y[i] for i in A_idx])

                # Step 2: Transform Xtr using (σ^2 C)^(-1/2)
                C = es.sm.C
                sigma = es.sigma
                try:
                    cov_inv_sqrt = np.linalg.inv(np.sqrt(sigma ** 2 * C))
                except np.linalg.LinAlgError:
                    cov_inv_sqrt = np.eye(dim)
                Xtr = (Xtr_raw - es.mean) @ cov_inv_sqrt.T

                # Step 3: Normalize ytr
                y_mean = np.mean(ytr_raw)
                y_std = np.std(ytr_raw)
                ytr = (ytr_raw - y_mean) / (y_std + 1e-8)

                # Step 4: Fit GP model
                try:
                    gp.fit(Xtr, ytr)
                    model_trained = True
                except Exception as e:
                    print(f"⚠️ GP fit failed: {e}")
                    model_trained = False

            else:
                model_trained = False

            # Step 5: Sample new population
            xs = es.ask()

            if model_trained:
                # Step 6: Evaluate using model
                X_scaled = (xs - es.mean) @ cov_inv_sqrt.T
                y_pred = gp.predict(X_scaled)

                # Step 7–8: Check if model is constant
                if (np.max(y_pred) - np.min(y_pred)) < min(1e-8, 0.05 * (np.max(ytr) - np.min(ytr))):
                    print("⚠️ GP surrogate marked as constant (not used)")
                    model_trained = False

            # Step 9: Evaluate real objective
            if model_trained:
                # Use real evaluation only on top half based on surrogate
                top_idx = np.argsort(y_pred)[:pop_size // 2]
                ys = []
                for i, x in enumerate(xs):
                    if i in top_idx:
                        y = problem(x)
                        ys.append(y)
                    else:
                        y = y_pred[i] * y_std + y_mean  # Rescale to original scale
                        ys.append(y)
            else:
                ys = [problem(x) for x in xs]

            # Step 10: Update archive and CMA-ES
            archive_X.extend(xs)
            archive_y.extend(ys)
            es.tell(xs, ys)

            print(f"[DTS-CMA-ES Iter {iteration}] f(best) = {np.min(ys):.5f}")
            if es.stop():
                break

        print("✅ DTS-CMA-ES Done. Final best:", es.result.xbest)

    # ----------------------------
    # 3) MF-GP-UCB (Multi-Fidelity GP)
    # ----------------------------
    @staticmethod
    def optimize_mf_gp_ucb(problem, dim, max_iter, pop_size):
        gp = Optimization.init_gp()
        scaler = StandardScaler()

        X_obs, y_obs = [], []

        for t in range(1, max_iter + 1):
            # Generování kandidátů kolem nejlepšího bodu
            if len(X_obs) > 0:
                center = X_obs[np.argmin(y_obs)]
                xs = center + np.random.normal(0, 0.3, size=(pop_size, dim))
            else:
                xs = np.random.uniform(-5, 5, size=(pop_size, dim))

            # Pokud máme dostatek dat, využijeme GP pro UCB
            if len(X_obs) >= pop_size:
                X_scaled = scaler.fit_transform(np.array(X_obs))
                gp.fit(X_scaled, np.array(y_obs))

                X_test = scaler.transform(xs)
                mu, sigma = gp.predict(X_test, return_std=True)

                # Beta_t podle Srinivas et al. (2010)
                beta_t = 2 * np.log(len(X_obs) + 1)
                ucb = mu - np.sqrt(beta_t) * sigma

                # Vybereme nejlepší kandidáty podle UCB
                best_idx = np.argmin(ucb)
                x_sel = xs[best_idx]
            else:
                # Inicializace náhodným bodem
                x_sel = xs[np.random.randint(len(xs))]

            # Hodnotíme funkci (plná věrnost)
            y_sel = problem(x_sel)

            X_obs.append(x_sel)
            y_obs.append(y_sel)

            print(f"[MF-GP-UCB Iter {t}] f(x)={y_sel:.5f}")

        print("✅ MF-GP-UCB Done. Best:", X_obs[np.argmin(y_obs)])

    # ----------------------------
    # 4) ES-CMA-ES (Ensemble Surrogates)
    # ----------------------------
    @staticmethod
    def optimize_es_cma_es(problem, dim, max_iter, pop_size):
        # Parametry
        lambda_pre = 3 * pop_size
        lambda_c = pop_size
        max_lambda_c = int(0.5 * pop_size)
        max_no_improve = 5
        no_improve_counter = 0
        ilcb_kappa = 1.0

        # 1. Inicializace – LHS sampling v [0, 1]^D
        lhs = qmc.LatinHypercube(d=dim)
        X_init = lhs.random(n=lambda_pre)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_init)
        y_init = [problem(scaler.inverse_transform([x])[0]) for x in X_scaled]

        # 2. Trénuj počáteční GP
        gp = Optimization.init_gp()
        gp.fit(X_scaled, y_init)

        # 3. Najdi nejlepší bod jako start CMA-ES
        x0 = scaler.inverse_transform([X_scaled[np.argmin(y_init)]])[0]
        es = cma.CMAEvolutionStrategy(x0, 1.0, {'popsize': pop_size})

        # 4. Archiv
        archive_X = list(X_scaled)
        archive_y = list(y_init)
        best_f = np.min(y_init)

        for iteration in range(max_iter):
            # 5. Získej kandidáty z CMA-ES
            candidates = es.ask()
            candidates_scaled = scaler.transform(candidates)

            # 6. ILCB = μ - κ·σ
            def ilcb(x_scaled):
                mu, std = gp.predict([x_scaled], return_std=True)
                return mu[0] - ilcb_kappa * std[0]

            ilcb_values = [ilcb(x) for x in candidates_scaled]

            # 7. Vyber top pop_size bodů
            top_idx = np.argsort(ilcb_values)[:pop_size//2]
            print(f"{len(ilcb_values)},   {pop_size}")
            selected_candidates = [candidates[i] for i in top_idx]

            # 8. Vyhodnoť vybrané body
            y_selected = [problem(x) for x in selected_candidates]

            # 9. Aktualizuj archiv a GP
            archive_X.extend(scaler.transform(selected_candidates))
            archive_y.extend(y_selected)
            gp.fit(archive_X, archive_y)

            # 10. Update CMA-ES
            es.tell(selected_candidates, y_selected)

            # 11. λ_c adaptace
            f_iter_best = np.min(y_selected)
            if f_iter_best < best_f:
                best_f = f_iter_best
                no_improve_counter = 0
                lambda_c = min(lambda_c + 1, max_lambda_c)
            else:
                no_improve_counter += 1
                if no_improve_counter >= max_no_improve:
                    lambda_c = pop_size  # reset
                    no_improve_counter = 0

            print(f"[ES-CMA-ES Iter {iteration}] f(best) = {best_f:.5f}")
            if es.stop():
                break

        print("✅ ES-CMA-ES Done. Best:", es.result.xbest)

    # ----------------------------
    # 5) GA-Galapagos baseline
    # ----------------------------
    @staticmethod
    def optimize_ga_galapagos(problem, dim, max_iter, pop_size):
        mu = pop_size // 2
        mutation_strength = 0.2
        max_mutation_strength = 1.0
        stagnation_counter = 0
        max_stagnation = 5

        # Initial population
        pop = np.random.randn(pop_size, dim)
        fitness = np.array([problem(ind) for ind in pop])
        best_fitness = np.min(fitness)
        best_solution = pop[np.argmin(fitness)]

        for iteration in range(max_iter):
            # Select top mu parents
            top_idx = np.argsort(fitness)[:mu]
            parents = pop[top_idx]

            # Diversity-aware mutation scaling
            diversity = np.std(parents, axis=0).mean()
            adaptive_strength = mutation_strength * (1 + diversity)

            # Generate offspring with Gaussian mutation
            offspring = parents + adaptive_strength * np.random.randn(*parents.shape)

            # Evaluate offspring
            offspring_fitness = np.array([problem(ind) for ind in offspring])

            # Combine and select next generation
            combined = np.vstack((parents, offspring))
            combined_fitness = np.hstack((fitness[top_idx], offspring_fitness))
            best_indices = np.argsort(combined_fitness)[:pop_size]
            pop = combined[best_indices]
            fitness = combined_fitness[best_indices]

            # Track global best
            current_best = np.min(fitness)
            if current_best < best_fitness:
                best_fitness = current_best
                best_solution = pop[np.argmin(fitness)]
                stagnation_counter = 0
                mutation_strength = max(0.2, mutation_strength * 0.8)
            else:
                stagnation_counter += 1
                if stagnation_counter >= max_stagnation:
                    mutation_strength = min(max_mutation_strength, mutation_strength * 1.2)
                    stagnation_counter = 0
                    print(f"⚠️ Stagnation detected. Mutation strength ↑ to {mutation_strength:.3f}")

            # CMA-ES exploitation every 10 iterations or on good descent
            if iteration % 10 == 0:
                print(f"(CMA-ES local search at iter {iteration})")
                es = cma.CMAEvolutionStrategy(best_solution, 0.5, {'popsize': pop_size})
                es.optimize(problem, iterations=20, verb_disp=0)
                refined = es.result.xbest
                refined_fit = problem(refined)
                if refined_fit < best_fitness:
                    best_fitness = refined_fit
                    best_solution = refined
                    pop[-1] = refined
                    fitness[-1] = refined_fit

            print(f"[GA-Galapagos Iter {iteration}] f(best)={best_fitness:.5f}")

        print("✅ GA-Galapagos Done. Best:", best_solution)
        return best_solution


    # ----------------------------
    # 6) Plain CMA-ES Optimization
    # ----------------------------
    @staticmethod
    def optimize_cma_es(problem, dim, max_iter, pop_size):
        es = cma.CMAEvolutionStrategy(dim * [0], 1.0, {'popsize': pop_size})
        for iteration in range(max_iter):
            xs = es.ask()
            ys = [problem(x) for x in xs]
            es.tell(xs, ys)
            print(f"[Iter {iteration}] f(best) = {min(ys):.5f}")
            if es.stop():
                break
        print("✅ Done. Final best solution:", es.result.xbest)

    # ----------------------------
    # 7) GP-assisted CMA-ES Optimization
    # ----------------------------
    @staticmethod
    def optimize_gp_cma_es(problem, dim, max_iter, pop_size):
        # CMA-ES initialization
        es = cma.CMAEvolutionStrategy(dim * [0], 1.0, {'popsize': pop_size})

        # Archive of evaluated solutions
        X_archive, y_archive = [], []

        # GP model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

        for gen in range(1, max_iter + 1):
            # 1. Ask CMA-ES for candidates
            X = es.ask()
            X = np.array(X)

            # 2. Train GP if we have enough data
            if len(X_archive) >= dim + 1:
                gp.fit(np.array(X_archive), np.array(y_archive))
                y_pred, sigma = gp.predict(X, return_std=True)
            else:
                # fallback: random predictions until GP has enough data
                y_pred = np.random.randn(len(X))
                sigma = np.ones(len(X))

            # 3. Select subset of candidates for true evaluation
            mu = pop_size // 2
            # Rank by GP mean (acquisition = mean - k*sigma could be used)
            idx = np.argsort(y_pred)[:mu]
            X_eval = X[idx]

            # 4. Evaluate expensive function
            y_eval = np.array([problem(x) for x in X_eval])

            # 5. Update archive
            X_archive.extend(X_eval.tolist())
            y_archive.extend(y_eval.tolist())

            # 6. Tell CMA-ES with true evaluated subset
            es.tell(X_eval.tolist(), y_eval.tolist())

            # 7. Print progress
            print(f"[Gen {gen}] best true f = {min(y_archive):.5f}")

            if es.stop():
                break

        print("✅ Done. Final best solution:", es.result.xbest)
        return es.result.xbest, min(y_archive)

    # ----------------------------
    # 8) Surrogate-Assisted Adaptive CMA-ES (SAACM-ES)
    # ----------------------------
    @staticmethod
    def optimize_saacm_es(problem, dim, max_iter, pop_size):
        # CMA-ES initialization
        es = cma.CMAEvolutionStrategy(dim * [0], 1.0, {'popsize': pop_size})

        # Archive of real evaluations
        X_archive, y_archive = [], []

        # Surrogate model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

        # Lifespan of surrogate usage (adaptive)
        surrogate_life = 1  # start with 1 gen
        surrogate_used = 0

        for gen in range(1, max_iter + 1):
            # 1. Generate population
            X = np.array(es.ask())

            # 2. If enough archive, train/update surrogate
            if len(X_archive) >= dim + 1:
                if surrogate_used >= surrogate_life:
                    gp.fit(np.array(X_archive), np.array(y_archive))
                    surrogate_used = 0

                # Surrogate prediction
                y_pred, sigma = gp.predict(X, return_std=True)

                # Select mu best according to surrogate
                mu = pop_size // 2
                idx = np.argsort(y_pred)[:mu]
                X_sel = X[idx]

                # Evaluate them with true objective
                y_sel = np.array([problem(x) for x in X_sel])

                # Measure surrogate error
                pred_sel = y_pred[idx]
                error = np.mean(np.abs(y_sel - pred_sel))

                # Adapt surrogate lifelength
                if error < 1e-2:   # good surrogate
                    surrogate_life = min(surrogate_life + 1, 10)
                else:              # bad surrogate
                    surrogate_life = max(surrogate_life // 2, 1)

                surrogate_used += 1

            else:
                # Initial bootstrap: evaluate whole population
                y_sel = np.array([problem(x) for x in X])
                X_sel = X

            # 3. Update archive
            X_archive.extend(X_sel.tolist())
            y_archive.extend(y_sel.tolist())

            # 4. Update CMA-ES
            es.tell(X_sel.tolist(), y_sel.tolist())

            # 5. Progress report
            print(f"[Gen {gen}] best f = {min(y_archive):.5f}, surrogate_life={surrogate_life}")

            if es.stop():
                break

        print("✅ Done. Final best solution:", es.result.xbest)
        return es.result.xbest, min(y_archive)

    # ----------------------------
    # 9) Adaptive CMA-ES (ACM-ES)
    # ----------------------------
    @staticmethod
    def optimize_acm_es(problem, dim, max_iter, pop_size):
        es = cma.CMAEvolutionStrategy(dim * [0], 1.0, {"popsize": pop_size})
        mu = pop_size // 2

        X_archive, y_archive = [], []
        surrogate = SVC(kernel="linear")

        for gen in range(1, max_iter + 1):
            X = np.array(es.ask())

            # --- Train surrogate if we have enough archive data ---
            if len(X_archive) >= 5:
                X_pairs, y_pairs = [], []
                for i in range(len(X_archive)):
                    for j in range(i + 1, len(X_archive)):
                        xi, xj = X_archive[i], X_archive[j]
                        yi, yj = y_archive[i], y_archive[j]
                        if yi == yj:
                            continue
                        X_pairs.append(xi - xj)
                        y_pairs.append(1 if yi < yj else 0)

                if len(X_pairs) > 0:
                    surrogate.fit(X_pairs, y_pairs)

                    # rank offspring by surrogate
                    scores = []
                    for x in X:
                        diffs = [x - xi for xi in X_archive]
                        preds = surrogate.decision_function(diffs)
                        scores.append(np.mean(preds))
                    ranked_idx = np.argsort(scores)
                    X = X[ranked_idx]

            # --- True evaluate top μ, surrogate-fill the rest ---
            y_true = np.array([problem(x) for x in X[:mu]])
            y_sur = np.zeros(pop_size - mu)

            if len(X_archive) >= 5:
                # surrogate predictions for rest
                for k, x in enumerate(X[mu:]):
                    diffs = [x - xi for xi in X_archive]
                    preds = surrogate.decision_function(diffs)
                    y_sur[k] = np.mean(preds)  # relative ranking score
            else:
                # fallback: random surrogate values
                y_sur = np.full(pop_size - mu, np.mean(y_true))

            y = np.concatenate([y_true, y_sur])

            # --- Update archive with true evaluations only ---
            X_archive.extend([np.array(x) for x in X[:mu]])
            y_archive.extend(y_true.tolist())

            # --- Tell CMA-ES with full λ values (true + surrogate) ---
            es.tell(X.tolist(), y.tolist())

            print(f"[Gen {gen}] best f = {min(y_true):.5f}")

            if es.stop():
                break

        print("✅ Done. Final best solution:", es.result.xbest)
        return es.result.xbest, min(y_archive)

    # ----------------------------
    # 10) Limited Covariance CMA-ES (LCC-CMA-ES)
    # ----------------------------
    @staticmethod
    def compute_state(go, sd, ah):
        # TODO: Implement features exactly from Table 1 in paper
        return np.concatenate([go, sd, ah])

    @staticmethod
    def decompose(action, dim):
        if action == 0:  # MiVD
            # TODO: Implement Minimum Variance Decomposition
            return [list(range(dim))]
        elif action == 1:  # RD
            perm = np.random.permutation(dim)
            return [perm[:dim//2], perm[dim//2:]]
        elif action == 2:  # MaVD
            # TODO: Implement Maximum Variance Decomposition
            return [list(range(dim))]
        else:
            raise ValueError("Invalid action")
    
    @staticmethod
    def optimize_lcc_cma_es(problem, dim, max_iter, pop_size):
        state_dim, action_dim = 10, 3  # placeholder sizes
        policy = ActorCritic(state_dim, action_dim)
        optimizer = optim.Adam(policy.parameters(), lr=3e-4)
        buffer = RolloutBuffer()

        # CMA-ES setup
        x0 = np.zeros(dim)
        sigma0 = 0.5
        es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize': pop_size})

        f0 = problem(x0)
        f_best, x_best = f0, x0

        gamma = 0.99  # discount factor
        eps_clip = 0.2

        for t in range(1, max_iter + 1):
            # ---- Build state ----
            state = Optimization.compute_state(np.zeros(3), np.zeros(3), np.zeros(4))
            state_t = torch.tensor(state, dtype=torch.float32)

            # ---- Agent acts ----
            action, log_prob, value = policy.act(state_t)

            # ---- Decomposition ----
            subgroups = Optimization.decompose(action, dim)

            # ---- Optimize each subgroup ----
            for group in subgroups:
                solutions = es.ask()
                fitness = [problem(x) for x in solutions]
                es.tell(solutions, fitness)
                idx_best = np.argmin(fitness)
                if fitness[idx_best] < f_best:
                    f_best, x_best = fitness[idx_best], solutions[idx_best]

            # ---- Reward ----
            reward = (f0 - f_best) / (abs(f0) + 1e-8)
            f0 = f_best

            # ---- Store ----
            buffer.store(state, action, log_prob, reward, value)

            print(f"[Iter {t}] f(x) = {f_best:.5f}, action={action}, reward={reward:.5f}")

            # ---- PPO Update ----
            if t % 10 == 0:
                states = torch.tensor(np.array(buffer.states), dtype=torch.float32)
                actions = torch.tensor(buffer.actions)
                old_log_probs = torch.stack(buffer.log_probs)
                rewards = torch.tensor(buffer.rewards, dtype=torch.float32)
                values = torch.tensor(buffer.values, dtype=torch.float32)

                # Compute discounted returns
                returns = []
                G = 0
                for r in reversed(rewards.tolist()):
                    G = r + gamma * G
                    returns.insert(0, G)
                returns = torch.tensor(returns, dtype=torch.float32)

                advantages = returns - values

                for _ in range(4):  # PPO epochs
                    log_probs, new_values, entropy = policy.evaluate(states, actions)
                    ratio = torch.exp(log_probs - old_log_probs)

                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    critic_loss = (returns - new_values).pow(2).mean()
                    entropy_bonus = entropy.mean()

                    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                buffer.clear()

        print("✅ Done. Final best solution:", x_best, "fitness =", f_best)
        return x_best, f_best

    # ----------------------------
    # 11) Meta Evolution Strategy (Meta-ES)
    # ----------------------------
    @staticmethod
    def run_solver(problem, dim, config, budget=50):
        x = np.random.randn(dim)
        best_f = problem(x)
        for _ in range(budget):
            offspring = [x + config.step_size * np.random.randn(dim) for _ in range(config.pop_size)]
            fitness = [problem(y) for y in offspring]
            idx = np.argmin(fitness)
            if fitness[idx] < best_f:
                best_f, x = fitness[idx], offspring[idx]
        return best_f
    
    @staticmethod
    def optimize_meta_es(problem, dim, max_iter, pop_size):
        """
        Meta-ES: optimalizace metaparametrů CMA-ES pomocí evoluční strategie
        s adaptivním učením krokové velikosti.
        """
        # Initialize meta-population (solvers)
        meta_population = [SolverConfig() for _ in range(pop_size)]
        best_config = None
        best_fitness = float("inf")

        for generation in range(1, max_iter + 1):
            # Evaluate each solver
            fitnesses = [Optimization.run_solver(problem, dim, cfg) for cfg in meta_population]

            # Track global best
            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_config = meta_population[gen_best_idx]

            print(f"[Gen {generation}] best f = {best_fitness:.5f} using solver {vars(best_config)}")

            # Selection (tournament)
            selected = []
            for _ in range(pop_size):
                i, j = random.sample(range(pop_size), 2)
                winner = meta_population[i] if fitnesses[i] < fitnesses[j] else meta_population[j]
                selected.append(winner)

            # Create next generation
            new_population = []
            for i in range(0, pop_size, 2):
                parent1, parent2 = selected[i], selected[(i + 1) % pop_size]
                child1 = parent1.crossover(parent2)
                child2 = parent2.crossover(parent1)
                child1.mutate()
                child2.mutate()
                new_population.extend([child1, child2])

            meta_population = new_population[:pop_size]

        print("✅ Done. Best solver config:", vars(best_config), "achieved fitness:", best_fitness)
        return best_config, best_fitness

    # ----------------------------
    # 12) CMA-ES with Latent Encoding Decoding (CMA-ES-LED)
    # ----------------------------
    @staticmethod
    def optimize_cma_es_led(problem, dim, max_iter, pop_size):
        # Strategy parameters
        lambda_ = pop_size
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)

        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        # Initialization
        m = np.zeros(dim)          # initial mean
        sigma = 0.3                # initial step-size
        pc = np.zeros(dim)         # evolution path for covariance
        ps = np.zeros(dim)         # evolution path for sigma
        B = np.eye(dim)            # eigenbasis
        D = np.ones(dim)           # scaling
        C = np.eye(dim)            # covariance matrix
        inv_sqrt_C = np.eye(dim)   # C^{-1/2}
        eigeneval = 0
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x, best_f = None, np.inf

        for gen in range(1, max_iter + 1):
            # 1. Sample population
            arz = np.random.randn(lambda_, dim)
            ary = arz @ (B * D).T
            arx = m + sigma * ary

            # 2. Evaluate solutions
            fitness = np.array([problem(x) for x in arx])
            idx = np.argsort(fitness)
            arx, ary, arz, fitness = arx[idx], ary[idx], arz[idx], fitness[idx]

            # Track best
            if fitness[0] < best_f:
                best_f, best_x = fitness[0], arx[0].copy()

            # 3. Weighted recombination
            y_w = np.dot(weights, ary[:mu])
            m += sigma * y_w

            # 4. Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (inv_sqrt_C @ y_w)
            norm_ps = np.linalg.norm(ps)
            hsig = int(norm_ps / np.sqrt(1 - (1 - cs) ** (2 * gen)) / chiN < (1.4 + 2 / (dim + 1)))

            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * y_w

            # 5. Covariance update (with LED mask)
            artmp = ary[:mu]
            delta_hsig = (1 - hsig) * cc * (2 - cc)

            # Compute LED mask (low effective dimensions)
            snr = np.abs(y_w) / (np.sqrt(np.diag(C)) + 1e-12)
            threshold = np.quantile(snr, 0.5)  # q = 0.5 as in paper
            mask = (snr >= threshold).astype(float)
            M = np.diag(mask)

            C = (1 - c1 - cmu) * C \
                + c1 * (np.outer(pc, pc) + delta_hsig * C) \
                + cmu * artmp.T @ np.diag(weights) @ artmp * M

            # 6. Step-size control
            sigma *= np.exp((cs / damps) * (norm_ps / chiN - 1))

            # 7. Update B, D from C
            if gen - eigeneval > lambda_ / (c1 + cmu) / dim / 10:
                eigeneval = gen
                C = np.triu(C) + np.triu(C, 1).T
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-30))
                inv_sqrt_C = B @ np.diag(D ** -1) @ B.T

            print(f"[Gen {gen}] f(x) = {best_f:.5f}, eff_dim = {int(np.sum(mask))}")

        print("✅ Done. Final best solution:", best_x, "fitness =", best_f)
        return best_x, best_f
        
    # ----------------------------
    # 13) LLAMA-ES (Low-Latency Adaptive Multi-Armed ES)
    # ----------------------------
    @staticmethod
    def query_llama(history_text):
        """
        Query LLaMA-3 exactly as in Kramer (2024), using the Appendix prompt template.
        """
        prompt = f"""
    You are a CMA-ES hyperparameter tuning expert.
    Given the last 10 iterations of CMA-ES, propose new hyperparameters.
    History (iteration, best fitness, sigma, population size):
    {history_text}

    Suggest new hyperparameters as a JSON object with keys:
    sigma, popsize, c1, cmu.
    Output only valid JSON.
    """

        response = openai.ChatCompletion.create(
            model="llama-3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.0,
        )

        text = response["choices"][0]["message"]["content"]
        try:
            params = eval(text)  # or json.loads if API outputs strict JSON
        except Exception:
            params = {"sigma": 1.0, "popsize": 10, "c1": 2.0/(dim**2), "cmu": 1.0/np.log(dim+1)}
        return params
    
    @staticmethod
    def optimize_llama_es(problem, dim, max_iter, pop_size):
        """
        Exact LLaMA-ES implementation according to Kramer (2024).
        - CMA-ES inner loop
        - Logging of recent history
        - Prompting LLaMA-3 with Appendix template
        - Restarting CMA-ES with suggested hyperparameters
        """
        x0 = np.zeros(dim)
        sigma = 1.0
        es = cma.CMAEvolutionStrategy(x0, sigma, {"popsize": pop_size})

        best_val, best_x = np.inf, None
        history = []
        iteration = 0

        while iteration < max_iter and not es.stop():
            # Phase: run CMA-ES for 10 iterations
            for _ in range(10):
                if iteration >= max_iter:
                    break
                solutions = es.ask()
                values = [problem(x) for x in solutions]
                es.tell(solutions, values)

                fmin = min(values)
                if fmin < best_val:
                    best_val, best_x = fmin, solutions[np.argmin(values)]

                history.append((iteration, fmin, es.sigma, es.popsize))
                iteration += 1

            # Format last 10 iterations as text for LLaMA prompt
            history_text = "\n".join(
                [f"{h[0]} {h[1]:.6f} {h[2]:.6f} {h[3]}" for h in history[-10:]]
            )

            # Ask LLaMA-3 for new hyperparameters
            params = Optimization.query_llama(history_text)

            # Restart CMA-ES with new hyperparameters
            es = cma.CMAEvolutionStrategy(
                best_x if best_x is not None else x0,
                params.get("sigma", es.sigma),
                {
                    "popsize": int(params.get("popsize", es.popsize)),
                    "c1": params.get("c1", es.opts.get("c1")),
                    "cmu": params.get("cmu", es.opts.get("cmu")),
                },
            )
            print(f"[LLaMA-ES] Restart with {params}, best f = {best_val:.5f}")

        print("✅ Done. Final best solution:", best_x, "fitness =", best_val)
        return best_x, best_val

    # ----------------------------
    # 14) RBF surrogate CMA-ES
    # ----------------------------
    @staticmethod
    def optimize_rbf_cma_es(problem, dim, max_iter, pop_size):
        # CMA-ES initialization
        sigma = 0.5
        es = cma.CMAEvolutionStrategy(dim * [0], sigma, {'popsize': pop_size})

        # Archive of true evaluations
        archive_X, archive_y = [], []

        # Surrogate parameters
        n_g = 1      # surrogate generations before true evaluation
        max_ng = 10  # cap as in the paper
        rbf = RBFModel()

        # Tracking best solution
        best_x, best_f = None, np.inf

        gen = 0
        while not es.stop() and gen < max_iter:
            # Step 1: get CMA-ES population
            X = es.ask()

            if len(archive_X) >= dim + 1:
                # Step 2: Surrogate phase
                rbf.fit(archive_X, archive_y)
                y_pred = [rbf.predict(x)[0] for x in X]
                es.tell(X, y_pred)
                es.disp()
                gen += 1
                print(f"[Gen {gen}] Surrogate evaluation, best pred = {min(y_pred):.5f}")
            else:
                # Step 3: Initial true evaluation
                y_true = [problem(x) for x in X]
                es.tell(X, y_true)
                archive_X.extend(X)
                archive_y.extend(y_true)
                es.disp()
                gen += 1
                f_min = min(y_true)
                if f_min < best_f:
                    best_f = f_min
                    best_x = X[np.argmin(y_true)]
                print(f"[Gen {gen}] True evaluation, best f = {best_f:.5f}")
                continue

            # Step 4: Every n_g generations, do a true evaluation
            if gen % n_g == 0:
                X_true = es.ask()
                y_true = [problem(x) for x in X_true]
                es.tell(X_true, y_true)

                # Update archive
                archive_X.extend(X_true)
                archive_y.extend(y_true)

                f_min = min(y_true)
                if f_min < best_f:
                    best_f = f_min
                    best_x = X_true[np.argmin(y_true)]

                # --- Surrogate error estimation (Algorithm 1, Step 8) ---
                y_pred = [rbf.predict(x)[0] for x in X_true]
                err = np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

                # Adapt n_g (Algorithm 1, Step 9)
                if err < 1e-3 and n_g < max_ng:
                    n_g += 1
                elif err > 1.0 and n_g > 1:
                    n_g = max(1, n_g // 2)

                print(f"[Gen {gen}] True evaluation, best f = {best_f:.5f}, surrogate err = {err:.3e}, n_g = {n_g}")

        print("✅ Done. Final best solution:", best_x, "fitness =", best_f)
        return best_x, best_f

    # ----------------------------
    # 15) SVM surrogate CMA-ES
    # ----------------------------
    @staticmethod
    def optimize_svm_cma_es(problem, dim, max_iter, pop_size, g_start=5, gm=3, gm_lambda=1.5):
        # Initialize CMAES
        # CMA-ES initialization
        sigma0 = 0.5
        es = cma.CMAEvolutionStrategy(dim * [0], sigma0, {'popsize': pop_size})

        # archive
        archive_X, archive_y = [], []

        # surrogate
        svm = SVR(kernel="rbf", C=100, gamma="scale")

        best_x, best_f = None, np.inf

        # evolution control parameters (from paper)
        g_m = 5  # use surrogate every g_m-th generation
        mu = es.sp.mu

        for g in range(1, max_iter + 1):
            candidates = np.array(es.ask())

            if g % g_m == 0 and len(archive_X) >= pop_size:
                # surrogate generation
                X_train, y_train = np.array(archive_X), np.array(archive_y)
                svm.fit(X_train, y_train)

                y_pred = svm.predict(candidates)
                es.tell(candidates.tolist(), y_pred.tolist())

                # evaluate best μ candidates on true objective
                best_idx = np.argsort(y_pred)[:mu]
                y_true = [problem(candidates[i]) for i in best_idx]

                for i, val in zip(best_idx, y_true):
                    archive_X.append(candidates[i])
                    archive_y.append(val)

                gen_best = min(y_true)
            else:
                # normal CMA-ES generation with real evaluations
                y_true = [problem(x) for x in candidates]
                es.tell(candidates.tolist(), y_true)
                for x, val in zip(candidates, y_true):
                    archive_X.append(x)
                    archive_y.append(val)
                gen_best = min(y_true)

            if gen_best < best_f:
                best_f = gen_best
                best_x = candidates[np.argmin(y_true if g % g_m != 0 else np.argsort(y_pred)[:mu])]

            print(f"[Gen {g}] best f(x) = {best_f:.5f}")

            if es.stop():
                break

        return best_x, best_f

    # ----------------------------
    # 16) Quadratic regression CMA-ES
    # ----------------------------
    @staticmethod
    def optimize_lq_cma_es(problem, dim, max_iter, pop_size, k=20):
        """
        CMA-ES with linear/quadratic polynomial surrogate (lq-CMA-ES).
        """
        # CMA-ES setup
        sigma0 = 0.3
        es = cma.CMAEvolutionStrategy(dim * [0], sigma0, {"popsize": pop_size})

        # Archive of real evaluations
        archive_X, archive_y = [], []

        for gen in range(1, max_iter + 1):
            X = es.ask()

            y = []
            if len(archive_X) < k:
                # Not enough archive data → evaluate all
                for x in X:
                    fx = problem(x)
                    archive_X.append(x)
                    archive_y.append(fx)
                    y.append(fx)
            else:
                # Fit quadratic surrogate using k nearest neighbours
                archive_X_arr = np.array(archive_X)
                archive_y_arr = np.array(archive_y)

                y_pred = []
                for x in X:
                    dists = np.linalg.norm(archive_X_arr - x, axis=1)
                    idx = np.argsort(dists)[:k]

                    X_nn = archive_X_arr[idx]
                    y_nn = archive_y_arr[idx]

                    poly = PolynomialFeatures(degree=2, include_bias=True)
                    X_nn_poly = poly.fit_transform(X_nn)
                    model = LinearRegression().fit(X_nn_poly, y_nn)

                    x_poly = poly.transform([x])
                    y_hat = model.predict(x_poly)[0]
                    y_pred.append(y_hat)

                # Select μ best candidates to re-evaluate
                ranked_idx = np.argsort(y_pred)
                mu = es.sp.weights.mu
                reevaluate_idx = ranked_idx[:mu]

                for i, x in enumerate(X):
                    if i in reevaluate_idx:
                        fx = problem(x)
                        archive_X.append(x)
                        archive_y.append(fx)
                        y.append(fx)
                    else:
                        y.append(y_pred[i])

            es.tell(X, y)
            es.disp()

            best = es.best.f
            print(f"[Gen {gen}] best f(x) = {best}")

        return es.best.x, es.best.f

    # ----------------------------
    # 17) Linear-Quadratic CMA-ES
    # ----------------------------
    @staticmethod
    def optimize_lmm_cma_es(problem, dim, max_iter, pop_size, k_neighbors=10):
        # CMA-ES initialization
        sigma0 = 0.3
        es = cma.CMAEvolutionStrategy(dim * [0.0], sigma0, {'popsize': pop_size})

        # Archive of evaluated solutions
        archive_X = []
        archive_y = []

        for gen in range(1, max_iter + 1):
            X = es.ask()  # λ candidates

            # If archive too small, evaluate all candidates directly
            if len(archive_X) < k_neighbors + dim + 1:
                y = [problem(x) for x in X]
                archive_X.extend(X)
                archive_y.extend(y)
            else:
                # Surrogate prediction for each candidate
                y_pred = []
                for x in X:
                    # Mahalanobis transform with current CMA-ES covariance
                    m = np.array(es.mean)
                    C = np.array(es.sm.covariance_matrix)
                    sigma = es.sigma
                    Sigma = (sigma ** 2) * C

                    # Transform archive into Mahalanobis space
                    diff = np.array(archive_X) - x
                    M = inv(Sigma)
                    dists = np.sqrt(np.sum(diff @ M * diff, axis=1))

                    # Select k nearest neighbors
                    idx = np.argsort(dists)[:k_neighbors]
                    X_nn = np.array(archive_X)[idx]
                    y_nn = np.array(archive_y)[idx]

                    # Normalize inputs and outputs
                    scaler_X = StandardScaler()
                    X_scaled = scaler_X.fit_transform(X_nn)
                    y_scaled = (y_nn - np.mean(y_nn)) / (np.std(y_nn) + 1e-8)

                    # Build quadratic feature matrix [1, x, x^2, cross terms]
                    Phi = []
                    for xi in X_scaled:
                        phi = [1.0]
                        phi.extend(xi)
                        phi.extend(xi**2)
                        for i in range(len(xi)):
                            for j in range(i+1, len(xi)):
                                phi.append(xi[i]*xi[j])
                        Phi.append(phi)
                    Phi = np.array(Phi)

                    # Fit linear regression
                    reg = LinearRegression(fit_intercept=False)
                    reg.fit(Phi, y_scaled)

                    # Predict candidate
                    x_scaled = scaler_X.transform([x])[0]
                    phi_x = [1.0]
                    phi_x.extend(x_scaled)
                    phi_x.extend(x_scaled**2)
                    for i in range(len(x_scaled)):
                        for j in range(i+1, len(x_scaled)):
                            phi_x.append(x_scaled[i]*x_scaled[j])
                    phi_x = np.array(phi_x).reshape(1, -1)

                    y_hat_scaled = reg.predict(phi_x)[0]
                    y_hat = y_hat_scaled * (np.std(y_nn) + 1e-8) + np.mean(y_nn)
                    y_pred.append(y_hat)

                # Rank candidates by surrogate predictions
                ranked_idx = np.argsort(y_pred)
                mu = es.sp.weights.mu   # correct number of parents (integer)
                reevaluate_idx = ranked_idx[:mu]
                
                # Re-evaluate best μ candidates with true function
                y = []
                for i, x in enumerate(X):
                    if i in reevaluate_idx:
                        fx = problem(x)
                        archive_X.append(x)
                        archive_y.append(fx)
                        y.append(fx)
                    else:
                        y.append(y_pred[i])

            # Update CMA-ES with surrogate+true evaluated fitness values
            es.tell(X, y)
            es.disp()

            best = es.best
            print(f"[Gen {gen}] best f(x) = {best.f}")

        return es.best.x, es.best.f

    # ----------------------------
    # 18) Ensemble surrogate CMA-ES (NN + GP + RBF)
    # ----------------------------
    @staticmethod
    def optimize_ensemble_cma_es(problem, dim, max_iter, pop_size):
        def cv_mse(model, X, y, folds=5):
            if len(X) < folds:
                return np.inf
            kf = KFold(n_splits=folds, shuffle=True, random_state=0)
            mses = []
            for tr, va in kf.split(X):
                m = clone(model)   # ✅ instead of re-init with get_params()
                m.fit(X[tr], y[tr])
                mses.append(mean_squared_error(y[va], m.predict(X[va])))
            return np.mean(mses)

        # Initialize CMA-ES
        es = cma.CMAEvolutionStrategy(np.zeros(dim), 0.3, {'popsize': pop_size, 'verb_disp': 0})
        archive_X, archive_y = [], []

        FULL_REEVAL_EVERY = 10

        for gen in range(1, max_iter + 1):
            X = np.array(es.ask())
            n = len(X)

            y_true = np.array([problem(xi) for xi in X])
            archive_X.extend(X)
            archive_y.extend(y_true)

            if len(archive_X) >= dim + 1:
                X_train = np.array(archive_X)
                y_train = np.array(archive_y)

                scaler = StandardScaler().fit(X_train)
                Xs = scaler.transform(X_train)
                Xs_cand = scaler.transform(X)

                gp = GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(length_scale=np.ones(dim)) + WhiteKernel(),
                                              normalize_y=True, n_restarts_optimizer=1)
                rbf = Ridge(alpha=1.0)
                nn = MLPRegressor(hidden_layer_sizes=(20,), max_iter=300)

                models = [gp, rbf, nn]
                errors = [cv_mse(m, Xs, y_train) for m in models]
                errors = [max(e, 1e-12) for e in errors]

                weights = np.array([1.0 / e for e in errors])
                weights /= weights.sum()

                preds = []
                for m in models:
                    m.fit(Xs, y_train)
                    preds.append(m.predict(Xs_cand))
                ensemble_pred = np.average(np.stack(preds), axis=0, weights=weights)

                # Decide which points to reevaluate with true objective:
                if gen % FULL_REEVAL_EVERY == 0:
                    y_eval = ensemble_pred.copy()
                    for idx in range(n):
                        y_eval[idx] = problem(X[idx])
                else:
                    y_eval = ensemble_pred.copy()
                    topK = max(1, pop_size // 2)
                    idxs = np.argsort(ensemble_pred)[:topK]
                    for i in idxs:
                        y_eval[i] = problem(X[i])
                        archive_X.append(X[i]); archive_y.append(y_eval[i])

                es.tell(X, y_eval.tolist())
            else:
                es.tell(X, y_true.tolist())

            print(f"[Gen {gen}] best f(x) = {min(archive_y):.5f}")
            if es.stop():
                break

        return ['exdata\\ENSEMBLE-CMA-ES']

    # ----------------------------
    # 19) Transformer surrogate CMA-ES
    # ----------------------------
    @staticmethod
    def optimize_transformer_cma_es(problem, dim, max_iter, pop_size):
        es = cma.CMAEvolutionStrategy([0.0]*dim, 1.0, {'popsize': pop_size})
        surrogate = TransformerSurrogate(dim)
        optimizer = optim.Adam(surrogate.parameters(), lr=1e-3)
        scaler = StandardScaler()

        X_obs, y_obs = [], []
        lifelength_max = 3  # max generations to trust surrogate consecutively
        lifelength = 0
        error_threshold = 0.5  # RMSE threshold (normalized)
        min_evals_before_surrogate = 5

        for gen in range(max_iter):
            xs = np.asarray(es.ask(), dtype=float)
            ys = []

            # Determine if surrogate should be used
            use_surrogate = False
            if lifelength > 0:
                lifelength -= 1
                use_surrogate = True

            if lifelength == 0 and len(X_obs) >= min_evals_before_surrogate:
                # Train surrogate
                X_arr = np.asarray(X_obs)
                y_arr = np.asarray(y_obs)
                scaler.fit(X_arr)
                X_scaled = scaler.transform(X_arr)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                y_tensor = torch.tensor(y_arr, dtype=torch.float32)

                surrogate.train()
                for _ in range(100):
                    optimizer.zero_grad()
                    pred = surrogate(X_tensor)
                    loss = nn.MSELoss()(pred, y_tensor)
                    loss.backward()
                    optimizer.step()

                surrogate.eval()
                with torch.no_grad():
                    pred = surrogate(X_tensor).cpu().numpy()
                rmse = float(np.sqrt(mean_squared_error(y_arr, pred)))
                y_std = float(np.std(y_arr)) if y_arr.size > 1 else 1.0
                nrmse = rmse / max(y_std, 1e-12)

                if nrmse < error_threshold:
                    lifelength = lifelength_max
                    use_surrogate = True
                else:
                    use_surrogate = False

            for xi in xs:
                if use_surrogate:
                    x_scaled = scaler.transform(xi.reshape(1, -1))
                    x_t = torch.tensor(x_scaled, dtype=torch.float32)
                    with torch.no_grad():
                        yi = float(surrogate(x_t).cpu().numpy())
                    # Optionally, distance check (trust-region) could be added
                else:
                    yi = float(problem(xi))
                ys.append(yi)

            X_obs.extend(xs.tolist())
            y_obs.extend(ys)
            es.tell(xs, ys)

            print(f"[Gen {gen}] best f(x) = {np.min(ys):.5f}, "
                  f"{'SURROGATE' if use_surrogate else 'TRUE'}, "
                  f"nRMSE={'{:.3f}'.format(nrmse) if 'nrmse' in locals() else 'N/A'}")

            if es.stop():
                break

        print("Done. Best:", es.result.xbest, "fitness =", es.result.fbest)
        return es.result

    # ----------------------------
    # 20) HNN RL CMA-ES
    # ----------------------------
    @staticmethod
    def optimize_hnn_rl_cma_es(problem, dim, max_iter, pop_size):
        # ---------- helpers ----------
        def eval_true_batch(X):
            ys = []
            for xi in X:
                try:
                    yi = float(problem(xi))
                except Exception:
                    # COCO/cocoex: problem.evaluate(np.array)
                    yi = float(problem.evaluate(np.asarray(xi, dtype=float)))
                ys.append(yi)
            return np.array(ys, dtype=float)

        def cv_nrmse(model, scaler, X, y, k=5):
            """k-fold CV nRMSE (normalized by std(y))."""
            if len(X) < k + 2 or np.std(y) == 0.0:
                return np.inf
            kf = KFold(n_splits=min(k, len(X)))
            errs = []
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
            for tr, te in kf.split(X):
                scaler_ = StandardScaler().fit(X[tr])
                Xm_tr = scaler_.transform(X[tr])
                Xm_te = scaler_.transform(X[te])
                m_ = MLPRegressor(
                    hidden_layer_sizes=(64, 64),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=1e-3,
                    max_iter=500,
                    random_state=0,
                )
                m_.fit(Xm_tr, y[tr])
                y_hat = m_.predict(Xm_te)
                errs.append(mean_squared_error(y[te], y_hat))
            rmse = float(np.sqrt(np.mean(errs)))
            return rmse / (np.std(y) + 1e-12)

        # ---------- simple tabular ε-greedy Q-learning ----------
        class QSwitch:
            # actions: 0=REAL_ONLY, 1=PRESELECTx5, 2=PRESELECTx10
            def __init__(self, eps_start=0.2, eps_end=0.01, eps_decay=0.995, alpha=0.3, gamma=0.6):
                self.Q = {}  # dict[(b_err,b_prog,b_sigma)] -> [Q0,Q1,Q2]
                self.eps = eps_start
                self.eps_end = eps_end
                self.eps_decay = eps_decay
                self.alpha = alpha
                self.gamma = gamma

            @staticmethod
            def _bin(x, cuts):
                # place x in bin 0..len(cuts)
                return int(np.digitize([x], cuts)[0])

            def _state_key(self, nrmse, prog, log_sigma):
                # three coarse bins per feature (tune as needed)
                be = self._bin(nrmse, [0.15, 0.30])      # error small/med/large
                bp = self._bin(prog,  [1e-3, 1e-2])     # progress small/med/large
                bs = self._bin(log_sigma, [-2.0, 0.0])  # step-size small/med/large
                return (be, bp, bs)

            def act(self, nrmse, prog, log_sigma):
                s = self._state_key(nrmse, prog, log_sigma)
                if s not in self.Q:
                    self.Q[s] = [0.0, 0.0, 0.0]
                if np.random.rand() < self.eps:
                    return np.random.randint(0, 3), s
                q = self.Q[s]
                return int(np.argmax(q)), s

            def update(self, s, a, reward, s_next):
                if s not in self.Q:
                    self.Q[s] = [0.0, 0.0, 0.0]
                if s_next not in self.Q:
                    self.Q[s_next] = [0.0, 0.0, 0.0]
                qsa = self.Q[s][a]
                td_target = reward + self.gamma * max(self.Q[s_next])
                self.Q[s][a] = qsa + self.alpha * (td_target - qsa)
                self.eps = max(self.eps_end, self.eps * self.eps_decay)

        # ---------- setup ----------
        es = cma.CMAEvolutionStrategy(dim * [0.0], 1.0, {'popsize': pop_size})
        scaler = StandardScaler()
        surrogate = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=800,
            random_state=0,
        )

        X_obs, y_obs = [], []
        warmup_gens = 3
        # thresholds for model usage
        nrmse_gate = 0.35         # allow model only if nRMSE <= this
        # oversampling choices for preselection
        oversampling = {0: 1, 1: 5, 2: 10}

        agent = QSwitch()
        best_so_far = np.inf
        last_best = np.inf

        # ---------- main loop ----------
        for gen in range(max_iter):
            # 1) propose λ candidates with CMA (we might oversample later)
            X_pop = np.array(es.ask(), dtype=float)  # shape (λ, d)

            # 2) warm-up or surrogate CV error
            mode = "TRUE"
            nrmse = np.nan
            overs = 1

            if len(X_obs) >= max(pop_size, 2 * dim):
                # fit scaler/surrogate on all observed data
                scaler.fit(np.asarray(X_obs, dtype=float))
                Xs = scaler.transform(np.asarray(X_obs, dtype=float))
                surrogate.fit(Xs, np.asarray(y_obs, dtype=float))

                # estimate nRMSE by CV
                nrmse = cv_nrmse(surrogate, scaler, X_obs, y_obs, k=5)

                if gen >= warmup_gens and np.isfinite(nrmse) and nrmse <= nrmse_gate:
                    # RL decides how much preselection to use
                    # progress metric: relative improvement of best over last gen
                    prog = max(0.0, (last_best - best_so_far) / (abs(last_best) + 1e-12))
                    action, s_key = agent.act(nrmse, prog, np.log(es.sigma + 1e-12))
                    overs = oversampling[action]
                    mode = "SURRO-PRESEL" if action > 0 else "TRUE"
                else:
                    mode = "TRUE"
            else:
                mode = "TRUE"

            # 3) if using preselection, oversample, score with surrogate, keep best λ
            if mode == "SURRO-PRESEL":
                lam = len(X_pop)
                # oversample with current mean/cov; CMA lets us ask more
                X_over = np.array([es.ask(1)[0] for _ in range((overs - 1) * lam)], dtype=float)
                X_all = np.vstack([X_pop, X_over])
                # score with surrogate
                X_all_scaled = scaler.transform(X_all)
                y_pred = surrogate.predict(X_all_scaled)
                # pick top λ by predicted *lowest* fitness
                idx = np.argsort(y_pred)[:lam]
                X_eval = X_all[idx]
            else:
                X_eval = X_pop

            # 4) evaluate true objective on selected λ and tell CMA
            y_eval = eval_true_batch(X_eval)
            es.tell(X_eval, y_eval.tolist())

            # 5) bookkeeping / dataset growth
            X_obs.extend(X_eval.tolist())
            y_obs.extend(y_eval.tolist())

            gen_best = float(np.min(y_eval))
            if gen == 0:
                last_best = gen_best
            best_so_far = min(best_so_far, gen_best)

            # reward for RL (observed only on gens we have both s and s_next)
            # encourage improvement and fewer true evals (small penalty per true eval)
            # We define "cost" = number of true evals (always λ here); penalty scaled small.
            cost_penalty = 0.001 * len(y_eval)
            reward = (last_best - best_so_far) - cost_penalty
            last_best = best_so_far

            # if we had an RL state this gen, bootstrap to the *next* state (roughly)
            if len(X_obs) >= max(pop_size, 2 * dim):
                # construct next state features for agent update
                next_prog = 0.0  # will be recomputed next iter; placeholder is fine for bootstrapping
                next_nrmse = nrmse if np.isfinite(nrmse) else 1.0
                s_next = agent._state_key(next_nrmse, next_prog, np.log(es.sigma + 1e-12))
                # If we acted this gen (i.e., trustworthy + not warmup), we have s_key
                if 's_key' in locals():
                    agent.update(s_key, 0 if overs == 1 else (1 if overs == 5 else 2), reward, s_next)

            # 6) logging
            nrmse_str = f"{nrmse:.3f}" if np.isfinite(nrmse) else "N/A"
            print(f"[Gen {gen}] best f(x) = {gen_best:.5f}, mode={mode}, nRMSE={nrmse_str}")

            if es.stop():
                break

        xbest = np.array(es.result.xbest, dtype=float)
        fbest = float(es.result.fbest)
        print(f"✅ Done. Best: {xbest} fitness = {fbest}")

    # ----------------------------
    # Register methods dynamically
    # ----------------------------
    @staticmethod
    def initialize():
        Optimization.methods = {
            "s-cma-es": Optimization.optimize_s_cma_es,
            "dts-cma-es": Optimization.optimize_dts_cma_es,
            "mf-gp-ucb": Optimization.optimize_mf_gp_ucb,
            "es-cma-es": Optimization.optimize_es_cma_es,
            "ga-galapagos": Optimization.optimize_ga_galapagos,
            "cma-es": Optimization.optimize_cma_es,
            "gp-cma-es": Optimization.optimize_gp_cma_es,
            "saacm-es": Optimization.optimize_saacm_es,
            "acm-es": Optimization.optimize_acm_es,
            "lcc-cma-es": Optimization.optimize_lcc_cma_es,
            "meta-es": Optimization.optimize_meta_es,
            "cma-es-led": Optimization.optimize_cma_es_led,
            "llama-es": Optimization.optimize_llama_es,
            "rbf-cma-es": Optimization.optimize_rbf_cma_es,
            "svm-cma-es": Optimization.optimize_svm_cma_es,
            "lq-cma-es": Optimization.optimize_lq_cma_es,
            "lmm-cma-es": Optimization.optimize_lmm_cma_es,
            "ensemble-cma-es": Optimization.optimize_ensemble_cma_es,
            "transformer-cma-es": Optimization.optimize_transformer_cma_es,
            "hnn-rl-cma-es": Optimization.optimize_hnn_rl_cma_es,
        }

    @staticmethod
    def get_method(method_str):
        return Optimization.methods[method_str]
