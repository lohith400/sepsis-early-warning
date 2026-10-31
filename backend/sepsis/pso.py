"""Particle Swarm Optimisation with decaying inertia and a warm start."""
import numpy as np


def pso(fitness, v0, lo, hi, n_particles, n_iters, rng, noise, log=print):
    """Minimise ``fitness``. Particle 0 = v0, half the swarm near v0, the rest random.

    Returns (best vector, best loss, loss history, loss of v0)."""
    D = len(v0)
    vmax = 0.2 * (hi - lo)
    n_near = n_particles // 2
    X = np.vstack(
        [v0]
        + [np.clip(v0 + rng.normal(0, 1, D) * noise, lo, hi) for _ in range(n_near)]
        + [rng.uniform(lo, hi, D) for _ in range(n_particles - 1 - n_near)]
    )
    V = rng.uniform(-1, 1, (n_particles, D)) * vmax * 0.1
    pbest = X.copy()
    pfit = np.array([fitness(p) for p in X])
    gi = pfit.argmin()
    gbest, gfit = pbest[gi].copy(), float(pfit[gi])
    init_fit = float(fitness(v0))
    hist = [gfit]
    for it in range(n_iters):
        w = 0.9 - 0.5 * it / max(n_iters - 1, 1)          # inertia 0.9 -> 0.4
        r1, r2 = rng.random((n_particles, D)), rng.random((n_particles, D))
        V = np.clip(w * V + 1.5 * r1 * (pbest - X) + 1.5 * r2 * (gbest - X), -vmax, vmax)
        X = np.clip(X + V, lo, hi)
        for i in range(n_particles):
            f = fitness(X[i])
            if f < pfit[i]:
                pfit[i], pbest[i] = f, X[i].copy()
                if f < gfit:
                    gfit, gbest = float(f), X[i].copy()
        hist.append(gfit)
        if (it + 1) % 10 == 0:
            log(f"  PSO iteration {it + 1}/{n_iters} | best loss {gfit:.4f} (k-means start {init_fit:.4f})")
    return gbest, gfit, hist, init_fit
