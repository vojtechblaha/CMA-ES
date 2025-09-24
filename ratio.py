import numpy as np
from scipy.stats import kendalltau

class RealEvaluationRatio:
    def __init__(self, args):
        """
        args.real_evaluation_ratio = "{type}-{value}"
        """
        self.args = args

        # Init type
        self.type = args.real_evaluation_ratio.split("-")[0]

        # Init ratios
        if self.type == "static":
            self.ratios = []
            ratios_str = args.real_evaluation_ratio.split("-")[1]
            for ratio_str in ratios_str.split(","):
                self.ratios.append(float(ratio_str))
            self.ratio_index = 0
        elif self.type == "adaptive":
            self.gen = 0
            self.sur_gen = 1
            self.max_gen = 10
            self.waiting_update = False
            self.request_prediction = False
        elif self.type == "adaptive-saacm":
            self.g_start = 10
            self.n_max = 5
            self.T_err = 0.8
            self.beta_err = 0.2
            self.g = 0
            self.Err_tau = 0.5
            self.waiting_surrogates = 0

    def get(self):
        if self.type == "static":
            r = self.ratios[self.ratio_index]
            self.ratio_index = (self.ratio_index + 1) % len(self.ratios)
            return r
        elif self.type == "adaptive":
            if self.request_prediction:
                self.request_prediction = False
                return 0.0
            elif self.gen % self.sur_gen == 0:
                self.gen -= 1
                self.request_prediction = True
                self.waiting_update = True
                return 1.0
            else:
                return 0.0
        elif self.type == "adaptive-saacm":
            if self.g < self.g_start:
                return 1.0
            elif self.waiting_surrogate > 0:
                self.waiting_surrogate -= 1
                return 0.0
            elif self.waiting_surrogate == 0:
                frac = 1.0 - (self.Err_tau / self.T_err)
                self.waiting_surrogate = int(np.floor(np.clip(frac, 0.0, 1.0) * self.n_max))
                return 1.0
            

    def update(self, data):
        if self.type == "static":
            pass
        elif self.type == "adaptive":
            if self.waiting_update:
                err = np.mean((np.array(data['true']) - np.array(data['pred'])) ** 2)
                if err < 1e-3 and self.sur_gen < self.max_gen:
                    self.sur_gen += 1
                elif err > 1.0 and self.sur_gen > 1:
                    self.sur_gen = max(1, self.sur_gen // 2)
                self.waiting_update = False
            self.gen += 1
        elif self.type == "adaptive-saacm":
            tau, _ = kendalltau(np.array(data['true']), np.array(data['pred']))
            if np.isnan(tau):
                tau = 0.0
            err = 0.5 * (1.0 - tau)
            self.Err_tau = (1.0 - self.beta_err) * self.Err_tau + self.beta_err * err
            self.g += 1
