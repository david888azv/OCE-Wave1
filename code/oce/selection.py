"""Feature selection / regularization methods for OCE.

Each strategy receives the design matrix X (n_mol × n_feat), targets y, and a
list of feature keys, and returns:
    - subset of selected feature indices
    - fitted coefficients on the selected subset
    - intercept
    - leave-many-out CV RMSE

Strategies:
    A. Regularization (no combinatorial search)
       - ridge:        plain Ridge with optimized α (CV)
       - lasso:        L1; induces sparsity automatically
       - elastic_net:  L1+L2 combo
       - ard:          Bayesian ARD (Automatic Relevance Determination)

    B. Combinatorial search over binary masks
       - rfe:          Recursive Feature Elimination with CV
       - de:           Differential Evolution (binary, scipy.optimize)
       - sa:           Simulated Annealing (custom; one-bit-flip moves)
       - ga:           Genetic Algorithm (custom; baseline like in UNCLE)
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution
from sklearn.linear_model import (ARDRegression, ElasticNetCV, LassoCV,
                                  LinearRegression, Ridge, RidgeCV)
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, ShuffleSplit


def gc_weights(y: np.ndarray, formulas: list[str], T: float = 0.5) -> np.ndarray:
    """Garbulsky-Ceder-like Boltzmann weights w_i = exp(-(E_i - E_min_group)/T).

    Within each isomer group (same chemical formula) the lowest-energy
    structure gets weight 1 and higher-energy isomers decay exponentially.
    Singletons keep weight 1.  Translates the UNCLE GC ground-state-priority
    constraint into a soft sample-weight that any sklearn estimator accepts.
    """
    w = np.ones(len(y))
    groups: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(formulas):
        groups[f].append(i)
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        emin = min(y[i] for i in idxs)
        for i in idxs:
            w[i] = float(np.exp(-(y[i] - emin) / T))
    return w


@dataclass
class SelectionResult:
    name: str
    selected_idx: list[int]
    coef: np.ndarray              # only on selected features
    intercept: float
    cv_rmse: float
    fit_time_s: float
    n_features: int = field(init=False)

    def __post_init__(self):
        self.n_features = len(self.selected_idx)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_idx] @ self.coef + self.intercept


def _cv_rmse(X: np.ndarray, y: np.ndarray, idx: list[int],
             alpha: float = 1e-3, n_splits: int = 5, n_repeats: int = 8,
             seed: int = 0,
             sample_weight: np.ndarray | None = None) -> float:
    """Leave-many-out CV RMSE on a fixed feature subset.

    If sample_weight is given, the fit is weighted-Ridge and the test-set
    error is the weighted RMSE on the held-out fold.
    """
    if len(idx) == 0:
        return float(np.sqrt(np.mean((y - y.mean()) ** 2)))
    rng = np.random.default_rng(seed)
    rmses: list[float] = []
    n = X.shape[0]
    for r in range(n_repeats):
        ss = ShuffleSplit(n_splits=n_splits, test_size=max(2, n // 5),
                          random_state=int(rng.integers(2 ** 31)))
        for tr, te in ss.split(X):
            reg = Ridge(alpha=alpha, fit_intercept=True)
            sw_tr = None if sample_weight is None else sample_weight[tr]
            reg.fit(X[np.ix_(tr, idx)], y[tr], sample_weight=sw_tr)
            pred = reg.predict(X[np.ix_(te, idx)])
            err2 = (pred - y[te]) ** 2
            if sample_weight is not None:
                w_te = sample_weight[te]
                rmse = float(np.sqrt(np.sum(w_te * err2) /
                                     max(np.sum(w_te), 1e-12)))
            else:
                rmse = float(np.sqrt(np.mean(err2)))
            rmses.append(rmse)
    return float(np.mean(rmses))


# ---------- A. Regularization-only ----------

def fit_ridge_cv(X, y, alphas=None) -> SelectionResult:
    t0 = time.perf_counter()
    if alphas is None:
        alphas = np.logspace(-4, 2, 25)
    reg = RidgeCV(alphas=alphas, fit_intercept=True)
    reg.fit(X, y)
    idx = list(range(X.shape[1]))
    rmse = _cv_rmse(X, y, idx, alpha=float(reg.alpha_))
    return SelectionResult(
        name=f"Ridge(α={reg.alpha_:.3g})",
        selected_idx=idx,
        coef=np.asarray(reg.coef_),
        intercept=float(reg.intercept_),
        cv_rmse=rmse,
        fit_time_s=time.perf_counter() - t0,
    )


def fit_lasso(X, y, n_alphas=80) -> SelectionResult:
    t0 = time.perf_counter()
    n_splits = min(5, X.shape[0] - 1)
    reg = LassoCV(n_alphas=n_alphas, cv=n_splits, max_iter=20000,
                  fit_intercept=True, n_jobs=1)
    reg.fit(X, y)
    nonzero = [i for i, c in enumerate(reg.coef_) if abs(c) > 1e-10]
    coef = np.array([reg.coef_[i] for i in nonzero])
    rmse = _cv_rmse(X, y, nonzero) if nonzero else _cv_rmse(X, y, [])
    return SelectionResult(
        name=f"Lasso(α={reg.alpha_:.3g})",
        selected_idx=nonzero,
        coef=coef,
        intercept=float(reg.intercept_),
        cv_rmse=rmse,
        fit_time_s=time.perf_counter() - t0,
    )


def fit_elastic_net(X, y) -> SelectionResult:
    t0 = time.perf_counter()
    n_splits = min(5, X.shape[0] - 1)
    reg = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
                       n_alphas=30, cv=n_splits, max_iter=20000,
                       fit_intercept=True, n_jobs=1)
    reg.fit(X, y)
    nonzero = [i for i, c in enumerate(reg.coef_) if abs(c) > 1e-10]
    coef = np.array([reg.coef_[i] for i in nonzero])
    rmse = _cv_rmse(X, y, nonzero) if nonzero else _cv_rmse(X, y, [])
    return SelectionResult(
        name=f"ElasticNet(l1={reg.l1_ratio_:.2g}, α={reg.alpha_:.3g})",
        selected_idx=nonzero,
        coef=coef,
        intercept=float(reg.intercept_),
        cv_rmse=rmse,
        fit_time_s=time.perf_counter() - t0,
    )


def fit_ard(X, y, threshold=1e-2) -> SelectionResult:
    t0 = time.perf_counter()
    reg = ARDRegression(fit_intercept=True, threshold_lambda=1e6, max_iter=500)
    reg.fit(X, y)
    abs_coef = np.abs(reg.coef_)
    cutoff = max(threshold, abs_coef.max() * 1e-3)
    nonzero = [i for i, c in enumerate(abs_coef) if c > cutoff]
    coef = np.array([reg.coef_[i] for i in nonzero])
    rmse = _cv_rmse(X, y, nonzero) if nonzero else _cv_rmse(X, y, [])
    return SelectionResult(
        name="ARD",
        selected_idx=nonzero,
        coef=coef,
        intercept=float(reg.intercept_),
        cv_rmse=rmse,
        fit_time_s=time.perf_counter() - t0,
    )


# ---------- B. Combinatorial search ----------

def fit_rfe_cv(X, y, alpha: float = 1e-2) -> SelectionResult:
    t0 = time.perf_counter()
    n_splits = min(5, X.shape[0] - 1)
    reg = Ridge(alpha=alpha, fit_intercept=True)
    rfe = RFECV(reg, cv=n_splits, min_features_to_select=3, n_jobs=1)
    rfe.fit(X, y)
    idx = [i for i, ok in enumerate(rfe.support_) if ok]
    final = Ridge(alpha=alpha, fit_intercept=True).fit(X[:, idx], y)
    rmse = _cv_rmse(X, y, idx, alpha=alpha)
    return SelectionResult(
        name="RFE-CV(Ridge)",
        selected_idx=idx,
        coef=np.asarray(final.coef_),
        intercept=float(final.intercept_),
        cv_rmse=rmse,
        fit_time_s=time.perf_counter() - t0,
    )


def _binary_fitness(mask: np.ndarray, X, y, alpha: float, lam: float,
                    sample_weight: np.ndarray | None = None) -> float:
    """CV-RMSE penalized by feature-count (sparsity prior)."""
    idx = np.flatnonzero(mask).tolist()
    if not idx:
        return 1e6
    rmse = _cv_rmse(X, y, idx, alpha=alpha, n_splits=4, n_repeats=2,
                    sample_weight=sample_weight)
    return rmse + lam * len(idx)


def fit_de_binary(X, y, alpha: float = 1e-2, lam: float = 1e-3,
                  maxiter: int = 60, popsize: int = 20,
                  seed: int = 1) -> SelectionResult:
    """Binary Differential Evolution: continuous DE with sigmoid threshold."""
    t0 = time.perf_counter()
    n_feat = X.shape[1]

    def obj(x: np.ndarray) -> float:
        mask = x > 0.5
        return _binary_fitness(mask, X, y, alpha, lam)

    bounds = [(0.0, 1.0)] * n_feat
    res = differential_evolution(
        obj, bounds, maxiter=maxiter, popsize=popsize,
        mutation=(0.4, 1.0), recombination=0.7, seed=seed,
        polish=False, tol=1e-3, init="sobol", workers=1,
        updating="deferred",
    )
    mask = res.x > 0.5
    idx = np.flatnonzero(mask).tolist()
    if not idx:
        idx = [int(np.argmax(res.x))]
    final = Ridge(alpha=alpha, fit_intercept=True).fit(X[:, idx], y)
    rmse = _cv_rmse(X, y, idx, alpha=alpha)
    return SelectionResult(
        name=f"DE-binary",
        selected_idx=idx,
        coef=np.asarray(final.coef_),
        intercept=float(final.intercept_),
        cv_rmse=rmse,
        fit_time_s=time.perf_counter() - t0,
    )


def fit_sa(X, y, alpha: float = 1e-2, lam: float = 1e-3,
           steps: int = 1500, T0: float = 0.5, T_end: float = 1e-3,
           seed: int = 2,
           sample_weight: np.ndarray | None = None,
           name: str = "SA") -> SelectionResult:
    """Simulated Annealing on a binary mask. One-bit-flip moves."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n_feat = X.shape[1]
    mask = rng.random(n_feat) > 0.5
    if not mask.any():
        mask[rng.integers(n_feat)] = True
    score = _binary_fitness(mask, X, y, alpha, lam, sample_weight)
    best_mask, best_score = mask.copy(), score
    for s in range(steps):
        T = T0 * (T_end / T0) ** (s / max(1, steps - 1))
        flip = int(rng.integers(n_feat))
        mask[flip] = not mask[flip]
        if not mask.any():
            mask[flip] = not mask[flip]
            continue
        new_score = _binary_fitness(mask, X, y, alpha, lam, sample_weight)
        delta = new_score - score
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            score = new_score
            if score < best_score:
                best_mask, best_score = mask.copy(), score
        else:
            mask[flip] = not mask[flip]
    idx = np.flatnonzero(best_mask).tolist()
    final = Ridge(alpha=alpha, fit_intercept=True)
    final.fit(X[:, idx], y, sample_weight=sample_weight)
    rmse = _cv_rmse(X, y, idx, alpha=alpha, sample_weight=sample_weight)
    return SelectionResult(
        name=name,
        selected_idx=idx,
        coef=np.asarray(final.coef_),
        intercept=float(final.intercept_),
        cv_rmse=rmse,
        fit_time_s=time.perf_counter() - t0,
    )


def fit_ridge_gc(X, y, sample_weight: np.ndarray,
                 alphas=None) -> SelectionResult:
    """RidgeCV with Boltzmann-style sample weights (GC-soft)."""
    t0 = time.perf_counter()
    if alphas is None:
        alphas = np.logspace(-4, 2, 25)
    reg = RidgeCV(alphas=alphas, fit_intercept=True)
    reg.fit(X, y, sample_weight=sample_weight)
    idx = list(range(X.shape[1]))
    rmse = _cv_rmse(X, y, idx, alpha=float(reg.alpha_),
                    sample_weight=sample_weight)
    return SelectionResult(
        name=f"Ridge-GC(α={reg.alpha_:.3g})",
        selected_idx=idx,
        coef=np.asarray(reg.coef_),
        intercept=float(reg.intercept_),
        cv_rmse=rmse,
        fit_time_s=time.perf_counter() - t0,
    )


def fit_sa_gc(X, y, sample_weight: np.ndarray, **kw) -> SelectionResult:
    """SA selection with Boltzmann-style sample weights inside the CV."""
    return fit_sa(X, y, sample_weight=sample_weight, name="SA-GC", **kw)


def fit_rank_loss(X, y, formulas: list[str],
                  alpha: float = 1e-2, lambda_rank: float = 5.0,
                  margin: float = 0.0,
                  selected_idx: list[int] | None = None) -> SelectionResult:
    """Minimise squared loss + pairwise-hinge rank loss on isomer groups.

    Translates the Garbulsky-Ceder ε₁ constraint (preserve ordering of
    low-energy structures) into a soft loss:
        L(J,b) = Σ_i (ŷ_i - y_i)² + α‖J‖²
                 + λ_rank · Σ_{(i,j)∈pairs}  max(0, ŷ_i - ŷ_j + margin)²
    where 'pairs' are (i,j) with same formula and y_i < y_j (so we want
    ŷ_i < ŷ_j).  Solved with L-BFGS via scipy.

    If selected_idx is given, only those columns are used.  This lets the
    method be combined with prior selection (e.g. SA-then-rank-fit).
    """
    from scipy.optimize import minimize
    t0 = time.perf_counter()

    if selected_idx is None:
        Xs = X
        idx = list(range(X.shape[1]))
    else:
        Xs = X[:, selected_idx]
        idx = list(selected_idx)
    n, p = Xs.shape

    # Build pair list
    groups: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(formulas):
        groups[f].append(i)
    pairs: list[tuple[int, int]] = []
    for ids in groups.values():
        if len(ids) < 2:
            continue
        for a in ids:
            for b in ids:
                if y[a] < y[b]:
                    pairs.append((a, b))
    pairs_arr = np.array(pairs) if pairs else np.zeros((0, 2), dtype=int)

    # variables: J (p), b (1) -> packed as theta (p+1)
    def unpack(theta):
        return theta[:p], theta[p]

    def loss(theta):
        J, b = unpack(theta)
        yhat = Xs @ J + b
        res = yhat - y
        L = float(np.sum(res ** 2)) + alpha * float(np.sum(J ** 2))
        if pairs_arr.size > 0:
            ai = pairs_arr[:, 0]
            bi = pairs_arr[:, 1]
            d = yhat[ai] - yhat[bi] + margin
            viol = np.maximum(d, 0.0)
            L += lambda_rank * float(np.sum(viol ** 2))
        return L

    def grad(theta):
        J, b = unpack(theta)
        yhat = Xs @ J + b
        res = yhat - y
        gJ = 2.0 * Xs.T @ res + 2.0 * alpha * J
        gb = 2.0 * float(np.sum(res))
        if pairs_arr.size > 0:
            ai = pairs_arr[:, 0]
            bi = pairs_arr[:, 1]
            d = yhat[ai] - yhat[bi] + margin
            mask = d > 0
            if mask.any():
                d_act = d[mask]
                ai_act = ai[mask]
                bi_act = bi[mask]
                rows = Xs[ai_act] - Xs[bi_act]
                gJ += 2.0 * lambda_rank * (rows.T @ d_act)
                # b cancels in d so no contribution to gb
        g = np.empty(p + 1)
        g[:p] = gJ
        g[p] = gb
        return g

    # warm-start from Ridge
    init = Ridge(alpha=alpha, fit_intercept=True).fit(Xs, y)
    theta0 = np.concatenate([init.coef_, [init.intercept_]])
    res = minimize(loss, theta0, jac=grad, method="L-BFGS-B",
                   options=dict(maxiter=500, ftol=1e-9))
    Jfit, bfit = unpack(res.x)

    # sklearn-style rmse
    rmse = _cv_rmse(X, y, idx, alpha=alpha)
    return SelectionResult(
        name=f"RankLoss(λ={lambda_rank})",
        selected_idx=idx,
        coef=np.asarray(Jfit),
        intercept=float(bfit),
        cv_rmse=rmse,
        fit_time_s=time.perf_counter() - t0,
    )


def fit_sa_ensemble_rank(X, y, formulas: list[str], n_seeds: int = 5,
                          alpha: float = 1e-2, lambda_rank: float = 5.0,
                          margin: float = 0.0,
                          combine: str = "union",
                          sample_weight: np.ndarray | None = None
                          ) -> SelectionResult:
    """Multi-seed SA selection + rank-loss refit on the combined subset.

    'combine' = "union" takes any feature picked by any seed (richer subset);
    "intersection" keeps only features picked by ALL seeds (very sparse,
    high-confidence core).  After combination the rank-loss objective is
    used to learn coefficients with explicit isomer-ordering preservation.
    """
    runs = [fit_sa(X, y, seed=10 * s + 7, sample_weight=sample_weight)
            for s in range(n_seeds)]
    if combine == "union":
        idx = sorted(set().union(*[set(r.selected_idx) for r in runs]))
    elif combine == "intersection":
        common = set(runs[0].selected_idx)
        for r in runs[1:]:
            common &= set(r.selected_idx)
        idx = sorted(common)
        if not idx:  # fall back to union if intersection is empty
            idx = sorted(set().union(*[set(r.selected_idx) for r in runs]))
    elif combine.startswith("freq"):
        # combine = "freq2" means: keep features picked by at least 2 seeds
        thresh = int(combine[4:]) if len(combine) > 4 else 2
        counts: dict[int, int] = defaultdict(int)
        for r in runs:
            for i in r.selected_idx:
                counts[i] += 1
        idx = sorted(i for i, c in counts.items() if c >= thresh)
        if not idx:
            idx = sorted(counts.keys())
    elif combine == "best_cv":
        best = min(runs, key=lambda r: r.cv_rmse)
        idx = list(best.selected_idx)
    else:
        raise ValueError(f"Unknown combine={combine}")
    final = fit_rank_loss(X, y, formulas, alpha=alpha,
                          lambda_rank=lambda_rank, margin=margin,
                          selected_idx=idx)
    final.name = f"SA[{n_seeds}seeds,{combine}]+RankLoss(λ={lambda_rank})"
    return final


def fit_sa_multiseed(X, y, n_seeds: int = 5,
                     sample_weight: np.ndarray | None = None,
                     **kw) -> tuple[SelectionResult, dict]:
    """Run SA with several seeds; return the best result + variance stats.

    The 'best' result is selected by lowest CV-RMSE (post-selection).  Stats
    summarize the spread across seeds: mean ± std of n_features / cv_rmse,
    plus the Jaccard overlap of selected feature sets across seeds (a soft
    indicator of selection stability).
    """
    runs: list[SelectionResult] = []
    name_tag = "SA-GC" if sample_weight is not None else "SA"
    for s in range(n_seeds):
        runs.append(fit_sa(X, y, seed=10 * s + 7,
                           sample_weight=sample_weight,
                           name=name_tag, **kw))
    best = min(runs, key=lambda r: r.cv_rmse)
    n_feats = np.array([r.n_features for r in runs])
    rmses = np.array([r.cv_rmse for r in runs])
    sets = [set(r.selected_idx) for r in runs]
    jaccards = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            uni = sets[i] | sets[j]
            jaccards.append(len(sets[i] & sets[j]) / max(1, len(uni)))
    stats = dict(
        n_seeds=n_seeds,
        n_feat_mean=float(n_feats.mean()),
        n_feat_std=float(n_feats.std()),
        cv_rmse_mean=float(rmses.mean()),
        cv_rmse_std=float(rmses.std()),
        jaccard_mean=float(np.mean(jaccards)) if jaccards else 1.0,
        jaccard_std=float(np.std(jaccards)) if jaccards else 0.0,
    )
    return best, stats


def fit_ga(X, y, alpha: float = 1e-2, lam: float = 1e-3,
           pop: int = 30, gens: int = 60, mut_rate: float = 0.04,
           seed: int = 3) -> SelectionResult:
    """Simple genetic algorithm on binary masks (UNCLE-style baseline)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n_feat = X.shape[1]
    population = rng.random((pop, n_feat)) > 0.5
    for p in range(pop):
        if not population[p].any():
            population[p, rng.integers(n_feat)] = True
    fit = np.array([_binary_fitness(m, X, y, alpha, lam) for m in population])
    for g in range(gens):
        # tournament selection
        new = []
        for _ in range(pop):
            i, j = rng.integers(pop, size=2)
            new.append(population[i if fit[i] < fit[j] else j].copy())
        new = np.array(new)
        # uniform crossover (pairs)
        for k in range(0, pop - 1, 2):
            mask = rng.random(n_feat) < 0.5
            a, b = new[k].copy(), new[k + 1].copy()
            new[k][mask] = b[mask]
            new[k + 1][mask] = a[mask]
        # mutation
        flips = rng.random((pop, n_feat)) < mut_rate
        new = np.logical_xor(new, flips)
        for p in range(pop):
            if not new[p].any():
                new[p, rng.integers(n_feat)] = True
        new_fit = np.array([_binary_fitness(m, X, y, alpha, lam) for m in new])
        # elitism
        all_pop = np.vstack([population, new])
        all_fit = np.concatenate([fit, new_fit])
        order = np.argsort(all_fit)[:pop]
        population = all_pop[order]
        fit = all_fit[order]
    best = population[0]
    idx = np.flatnonzero(best).tolist()
    final = Ridge(alpha=alpha, fit_intercept=True).fit(X[:, idx], y)
    rmse = _cv_rmse(X, y, idx, alpha=alpha)
    return SelectionResult(
        name="GA",
        selected_idx=idx,
        coef=np.asarray(final.coef_),
        intercept=float(final.intercept_),
        cv_rmse=rmse,
        fit_time_s=time.perf_counter() - t0,
    )


METHODS = {
    "ridge":       fit_ridge_cv,
    "lasso":       fit_lasso,
    "elastic_net": fit_elastic_net,
    "ard":         fit_ard,
    "rfe":         fit_rfe_cv,
    # "de" (Differential Evolution binary) is implemented in fit_de_binary
    # but excluded from the default benchmark — its CV-RMSE is comparable
    # to GA but it costs ~10² × more wall-clock time.
    "sa":          fit_sa,
    "ga":          fit_ga,
}


if __name__ == "__main__":
    from pathlib import Path
    from oce.atomic_table import load_table
    from oce.dataset import load_dataset
    from oce.fit import build_design_matrix

    base = Path(__file__).resolve().parents[1]
    table = load_table(base / "data" / "atoms" / "atomic_table.json")
    train = load_dataset(base / "data" / "molecules" / "train.json")
    X, y, keys, _ = build_design_matrix(train, table)
    print(f"Design matrix: {X.shape},  features: {X.shape[1]}")

    for name, fn in METHODS.items():
        res = fn(X, y)
        print(f"\n{name:12s} → n_feat={res.n_features:4d}/{X.shape[1]}  "
              f"CV-RMSE={res.cv_rmse:.4f} eV  time={res.fit_time_s:.2f}s  "
              f"({res.name})")
