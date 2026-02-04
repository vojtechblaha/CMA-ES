import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ======================
# CONFIG
# ======================
RESULTS_DIR = "results"
ALGORITHMS = ["MY-CMA-ES", "LQ-CMA-ES"]

# regex na parsování posledního řádku
RE_NORMAL = re.compile(
    r"Mean evals:\s*([0-9.eE+-]+)\s*\+\-\s*([0-9.eE+-]+),\s*"
    r"Best FFTPS:\s*([0-9.eE+-]+)\s*\+\-\s*([0-9.eE+-]+)"
)
RE_INF = re.compile(
    r"Mean evals:\s*np\.inf.*?,\s*Best FFTPS:\s*([0-9.eE+-]+)\s*\+\-\s*([0-9.eE+-]+)"
)

# data[algorithm][function_id] = (mean_evals, std_evals, mean_fftps, std_fftps)
data = defaultdict(dict)

# ======================
# LOAD DATA
# ======================
for fname in os.listdir(RESULTS_DIR):
    for algo in ALGORITHMS:
        if fname.startswith(algo + "-"):
            try:
                func_id = int(fname.split("-")[-1].split(".")[0])
            except ValueError:
                continue

            path = os.path.join(RESULTS_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if "Mean evals" in l]

            if not lines:
                continue

            line = lines[-1].strip()

            m = RE_NORMAL.search(line)
            if m:
                mean_evals = float(m.group(1))
                std_evals  = float(m.group(2))
                mean_fftps = float(m.group(3))
                std_fftps  = float(m.group(4))
            else:
                m = RE_INF.search(line)
                if m:
                    mean_evals = 550.0
                    std_evals  = 0.0
                    mean_fftps = float(m.group(1))
                    std_fftps  = float(m.group(2))
                else:
                    continue  # řádek neznámého formátu

            data[algo][func_id] = (
                mean_evals,
                std_evals,
                mean_fftps,
                std_fftps,
            )

# ======================
# SORT FUNCTION IDS
# ======================
all_func_ids = sorted(
    set(fid for algo in data for fid in data[algo])
)

# ======================
# PLOT GRAPHS
# ======================
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)

# ======================
# PLOT 1: Mean evals
# ======================
ax = axes[0]

for algo in ALGORITHMS:
    xs, ys, yerr = [], [], []
    for fid in all_func_ids:
        if fid in data[algo]:
            mean_evals, std_evals, _, _ = data[algo][fid]
            xs.append(fid)
            ys.append(mean_evals)
            yerr.append(std_evals)

    ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4, label=algo)

ax.set_xlabel("Function ID")
ax.set_ylabel("Mean evals")
ax.set_title("Mean evals ± std per function")
ax.legend()
ax.grid(True)

# ======================
# PLOT 2: Best FFTPS
# ======================
ax = axes[1]

for algo in ALGORITHMS:
    xs, ys, yerr = [], [], []
    for fid in all_func_ids:
        if fid in data[algo]:
            _, _, mean_fftps, std_fftps = data[algo][fid]
            xs.append(fid)
            ys.append(mean_fftps)
            yerr.append(std_fftps)

    ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4, label=algo)

ax.set_xlabel("Function ID")
ax.set_ylabel("Best FFTPS")
ax.set_title("Best FFTPS ± std per function")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

