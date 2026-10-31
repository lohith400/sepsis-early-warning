"""Genetic Algorithm for feature selection (binary chromosome, wrapper fitness)."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from .metrics import auc_fast


def ga_select(Xa, ya, Xb, yb, cfg, rng, log=print, penalty: float = 0.004):
    """Choose between cfg.k_min and cfg.k_max features.

    Fitness = AUROC of a quick logistic regression (trained on A, judged on B) - penalty * n_features.
    Returns (mask, best-fitness history, mean-fitness history, quick AUROC of the chosen subset)."""
    nF = Xa.shape[1]
    cache = {}

    def fitness(m):
        key = m.tobytes()
        if key not in cache:
            cols = np.flatnonzero(m)
            lr = LogisticRegression(max_iter=100, class_weight="balanced").fit(Xa[:, cols], ya)
            a = auc_fast(yb, lr.decision_function(Xb[:, cols]))
            cache[key] = (a - penalty * len(cols), a)
        return cache[key]

    def repair(m):
        on = np.flatnonzero(m)
        if len(on) > cfg.k_max:
            m[rng.choice(on, len(on) - cfg.k_max, replace=False)] = False
        on = np.flatnonzero(m)
        if len(on) < cfg.k_min:
            m[rng.choice(np.flatnonzero(~m), cfg.k_min - len(on), replace=False)] = True
        return m

    def random_chromosome():
        m = np.zeros(nF, bool)
        m[rng.choice(nF, rng.integers(cfg.k_min, cfg.k_max + 1), replace=False)] = True
        return m

    pop = [random_chromosome() for _ in range(cfg.ga_pop)]
    best_hist, mean_hist = [], []
    for gen in range(cfg.ga_gen):
        fit = np.array([fitness(m)[0] for m in pop])
        best_hist.append(float(fit.max()))
        mean_hist.append(float(fit.mean()))
        order = np.argsort(-fit)
        new = [pop[order[0]].copy(), pop[order[1]].copy()]                         # elitism
        while len(new) < cfg.ga_pop:
            def pick():
                return pop[max(rng.choice(cfg.ga_pop, 3, replace=False), key=lambda i: fit[i])]  # tournament
            p1, p2 = pick(), pick()
            child = np.where(rng.random(nF) < 0.5, p1, p2) if rng.random() < 0.8 else p1.copy()  # crossover
            child = child ^ (rng.random(nF) < 0.04)                                            # mutation
            new.append(repair(child))
        pop = new
        if (gen + 1) % 5 == 0:
            log(f"  GA generation {gen + 1}/{cfg.ga_gen} | best fitness {fit.max():.4f}")
    best = pop[0]
    return best, best_hist, mean_hist, float(fitness(best)[1])
