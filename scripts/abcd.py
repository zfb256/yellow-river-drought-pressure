"""Monthly ABCD water-balance model (Thomas, 1981) with a temperature-index snow module."""
import numpy as np
from numba import njit

T_SNOW, T_RAIN = -1.0, 3.0
PARAM_BOUNDS = [(0.80, 1.0), (50.0, 1500.0), (0.0, 1.0), (0.001, 1.0)]


@njit(cache=True)
def run(p, pet, t, a, b, c, d, s0, g0, snow0=0.0, snow_trace=None):
    n = p.size
    q = np.empty(n)
    soil = np.empty(n)
    gw = np.empty(n)
    et = np.empty(n)
    s, g, snow = s0, g0, snow0
    for i in range(n):
        frac = min(max((t[i] - T_SNOW) / (T_RAIN - T_SNOW), 0.0), 1.0)
        snow += p[i] * (1.0 - frac)
        melt = snow * frac
        snow -= melt
        w = p[i] * frac + melt + s
        h = (w + b) / (2.0 * a)
        y = h - np.sqrt(max(h * h - w * b / a, 0.0))
        s = y * np.exp(-pet[i] / b)
        avail = w - y
        g = (g + c * avail) / (1.0 + d)
        q[i] = (1.0 - c) * avail + d * g
        soil[i] = s
        gw[i] = g
        et[i] = y - s
        if snow_trace is not None:
            snow_trace[i] = snow
    return q, soil, gw, et


def simulate(params, p, pet, t, spinup=10):
    """Run with a repeated-first-years spin-up so initial storages are in equilibrium."""
    a, b, c, d = params
    k = min(12 * spinup, p.size)
    reps = int(np.ceil(12 * spinup / k))
    sp = np.tile(p[:k], reps)
    se = np.tile(pet[:k], reps)
    st = np.tile(t[:k], reps)
    _, s_, g_, _ = run(sp, se, st, a, b, c, d, 0.5 * b, 0.0)
    return run(p, pet, t, a, b, c, d, s_[-1], g_[-1])


def kge(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[m], obs[m]
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def month_anomaly(x, month):
    out = np.full(x.shape, np.nan)
    for m in range(1, 13):
        k = month == m
        s = x[k]
        sd = np.nanstd(s)
        out[k] = (s - np.nanmean(s)) / sd if sd > 0 else 0.0
    return out


def calibrate(p, pet, t, q_ref, seed=0, month=None, q_anom_obs=None, w_obs=1.0):
    """Minimise 1 - KGE against q_ref, plus an optional observed-anomaly Spearman term."""
    from scipy.optimize import differential_evolution
    from scipy.stats import spearmanr

    has_obs = q_anom_obs is not None and np.isfinite(q_anom_obs).sum() >= 24

    def loss(x):
        q, *_ = simulate(x, p, pet, t)
        f = 1 - kge(q, q_ref)
        if has_obs:
            a = month_anomaly(q, month)
            m = np.isfinite(q_anom_obs) & np.isfinite(a)
            f += w_obs * (1 - spearmanr(a[m], q_anom_obs[m])[0])
        return f

    res = differential_evolution(loss, PARAM_BOUNDS, seed=seed, tol=1e-6, maxiter=300, polish=True)
    q, *_ = simulate(res.x, p, pet, t)
    return res.x, kge(q, q_ref)
