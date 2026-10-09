"""
Extended statistical analysis for the NBA Elo forecasting report.
Produces real numbers: bootstrap CIs, paired forecast tests, Brier (Murphy)
decomposition, ECE, and a from-scratch IRLS logistic-regression cross-check.
Run: python3 -I extended_stats.py /path/to/nbaallelo.csv
"""
import sys, json
import numpy as np
import pandas as pd

CSV = sys.argv[1]
rng = np.random.default_rng(2026)

df = pd.read_csv(CSV)
df = df[df["_iscopy"] == 0].copy()
df["date"] = pd.to_datetime(df["date_game"], format="%m/%d/%Y")
df = df.sort_values(["date", "gameorder"]).reset_index(drop=True)
df["y"] = (df["game_result"] == "W").astype(int)
df["hf"] = df["game_location"].map({"H": 1, "A": -1, "N": 0}).fillna(0).astype(int)

VAL = (1980, 2009); TEST = (2010, 2015)
K, HCA, REVERT = 20, 100, 0.25   # tuned previously


def run_elo(data, K, hca, revert, base=1500.0):
    ratings, last = {}, {}
    n = len(data)
    preds = np.empty(n); rdiff = np.empty(n)
    team = data["fran_id"].values; opp = data["opp_fran"].values
    hf = data["hf"].values; y = data["y"].values; yr = data["year_id"].values
    for i in range(n):
        t, o = team[i], opp[i]
        rt = ratings.get(t, base); ro = ratings.get(o, base)
        if revert > 0:
            if last.get(t, yr[i]) != yr[i]: rt = base + (1 - revert) * (rt - base)
            if last.get(o, yr[i]) != yr[i]: ro = base + (1 - revert) * (ro - base)
        last[t] = yr[i]; last[o] = yr[i]
        exp_t = 1.0 / (1.0 + 10 ** ((ro - rt - hca * hf[i]) / 400.0))
        preds[i] = exp_t; rdiff[i] = rt - ro
        ratings[t] = rt + K * (y[i] - exp_t)
        ratings[o] = ro + K * ((1 - y[i]) - (1 - exp_t))
    return preds, rdiff


p, rdiff = run_elo(df, K, HCA, REVERT)
yr = df["year_id"].values; y = df["y"].values; hf = df["hf"].values
tm = (yr >= TEST[0]) & (yr <= TEST[1])
vm = (yr >= VAL[0]) & (yr <= VAL[1])

yt = y[tm].astype(float)
p_elo = np.clip(p[tm], 1e-12, 1 - 1e-12)
f538 = df["forecast"].values[tm].astype(float)
f538 = np.clip(f538, 1e-12, 1 - 1e-12)
base_rate = df.loc[vm & (df.hf == 1), "y"].mean()
p_base = np.clip(np.where(hf[tm] == 1, base_rate, np.where(hf[tm] == -1, 1 - base_rate, 0.5)), 1e-12, 1-1e-12)

def acc(p, y): return float(np.mean((p >= 0.5) == (y == 1)))
def brier_vec(p, y): return (p - y) ** 2
def ll_vec(p, y): return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def point(p, y):
    return {"accuracy": acc(p, y), "brier": float(brier_vec(p, y).mean()),
            "log_loss": float(ll_vec(p, y).mean())}

res = {"n_test": int(tm.sum()), "models": {}}
for name, pp in [("elo", p_elo), ("base_rate", p_base), ("fivethirtyeight", f538)]:
    res["models"][name] = point(pp, yt)

# ---- Bootstrap CIs (percentile, 2000 resamples) ----
B = 2000; n = len(yt)
def bootstrap_ci(pp, yy, fn):
    stats = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        stats[b] = fn(pp[idx], yy[idx])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

res["bootstrap_ci_95"] = {}
for name, pp in [("elo", p_elo), ("base_rate", p_base), ("fivethirtyeight", f538)]:
    res["bootstrap_ci_95"][name] = {
        "accuracy": bootstrap_ci(pp, yt, acc),
        "brier": bootstrap_ci(pp, yt, lambda a, b: brier_vec(a, b).mean()),
        "log_loss": bootstrap_ci(pp, yt, lambda a, b: ll_vec(a, b).mean()),
    }

# ---- Paired comparisons: Diebold-Mariano style (mean diff / se) on per-game loss ----
def paired_test(loss_a, loss_b):
    d = loss_a - loss_b
    dbar = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    t = dbar / se
    # normal approx two-sided p
    from math import erfc, sqrt
    pval = erfc(abs(t) / sqrt(2))
    return {"mean_diff": float(dbar), "t_stat": float(t), "p_value": float(pval)}

res["paired_logloss"] = {
    "elo_vs_base": paired_test(ll_vec(p_elo, yt), ll_vec(p_base, yt)),
    "elo_vs_538": paired_test(ll_vec(p_elo, yt), ll_vec(f538, yt)),
}
res["paired_brier"] = {
    "elo_vs_base": paired_test(brier_vec(p_elo, yt), brier_vec(p_base, yt)),
    "elo_vs_538": paired_test(brier_vec(p_elo, yt), brier_vec(f538, yt)),
}

# ---- McNemar test: Elo vs 538 accuracy ----
elo_correct = (p_elo >= 0.5) == (yt == 1)
f_correct = (f538 >= 0.5) == (yt == 1)
b01 = int(np.sum(~elo_correct & f_correct))   # 538 right, elo wrong
b10 = int(np.sum(elo_correct & ~f_correct))   # elo right, 538 wrong
chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10) if (b01 + b10) > 0 else 0.0
from math import erfc, sqrt
res["mcnemar_elo_vs_538"] = {"b_elo_right_538_wrong": b10, "b_538_right_elo_wrong": b01,
                             "chi2_cc": float(chi2), "p_value": float(erfc(sqrt(chi2/2)) if chi2>0 else 1.0)}

# ---- Brier (Murphy) decomposition for Elo: BS = REL - RES + UNC ----
def murphy(p, y, nb=10):
    bins = np.linspace(0, 1, nb + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, nb - 1)
    ybar = y.mean()
    rel = res_ = 0.0
    table = []
    for b in range(nb):
        sel = idx == b
        nk = sel.sum()
        if nk == 0: continue
        pk = p[sel].mean(); ok = y[sel].mean()
        rel += nk * (pk - ok) ** 2
        res_ += nk * (ok - ybar) ** 2
        table.append({"bin": b, "n": int(nk), "pred_mean": float(pk), "obs_freq": float(ok)})
    N = len(y)
    rel /= N; res_ /= N; unc = ybar * (1 - ybar)
    return {"reliability": float(rel), "resolution": float(res_),
            "uncertainty": float(unc), "bs_check": float(rel - res_ + unc),
            "reliability_table": table}

res["brier_decomposition_elo"] = murphy(p_elo, yt)

# ---- Expected / Max Calibration Error for Elo ----
def calib_error(p, y, nb=10):
    bins = np.linspace(0, 1, nb + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, nb - 1)
    ece = mce = 0.0; N = len(y)
    for b in range(nb):
        sel = idx == b; nk = sel.sum()
        if nk == 0: continue
        gap = abs(p[sel].mean() - y[sel].mean())
        ece += nk / N * gap; mce = max(mce, gap)
    return {"ECE": float(ece), "MCE": float(mce)}

res["calibration_error_elo"] = calib_error(p_elo, yt)

# ---- From-scratch logistic regression via Newton-Raphson (IRLS) ----
# features: standardized pre-game rating difference, home field
def fit_logistic_irls(X, y, iters=50, tol=1e-10):
    n, k = X.shape
    beta = np.zeros(k)
    ll_hist = []
    for it in range(iters):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        mu = np.clip(mu, 1e-12, 1 - 1e-12)
        W = mu * (1 - mu)
        grad = X.T @ (y - mu)
        H = (X * W[:, None]).T @ X
        step = np.linalg.solve(H + 1e-8 * np.eye(k), grad)
        beta_new = beta + step
        ll = np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
        ll_hist.append(float(ll))
        if np.max(np.abs(step)) < tol:
            beta = beta_new; break
        beta = beta_new
    # covariance from inverse Fisher information
    eta = X @ beta; mu = np.clip(1/(1+np.exp(-eta)),1e-12,1-1e-12); W = mu*(1-mu)
    H = (X * W[:, None]).T @ X
    cov = np.linalg.inv(H + 1e-8*np.eye(X.shape[1]))
    se = np.sqrt(np.diag(cov))
    return beta, se, ll_hist, it + 1

# Build features on all games, split val/test
rd = rdiff.copy()
mu_rd = rd[vm].mean(); sd_rd = rd[vm].std()
X_all = np.column_stack([np.ones(len(rd)), (rd - mu_rd) / sd_rd, hf.astype(float)])
beta, se, ll_hist, n_iter = fit_logistic_irls(X_all[vm], y[vm].astype(float))
eta_t = X_all[tm] @ beta
p_logit = np.clip(1/(1+np.exp(-eta_t)), 1e-12, 1-1e-12)
res["logistic_irls"] = {
    "coefficients": {"intercept": float(beta[0]), "std_rating_diff": float(beta[1]), "home_field": float(beta[2])},
    "std_errors": {"intercept": float(se[0]), "std_rating_diff": float(se[1]), "home_field": float(se[2])},
    "z_stats": {"intercept": float(beta[0]/se[0]), "std_rating_diff": float(beta[1]/se[1]), "home_field": float(beta[2]/se[2])},
    "n_newton_iterations": int(n_iter),
    "loglik_path": ll_hist,
    "test_metrics": point(p_logit, yt),
}

# ---- Sensitivity grid: validation log-loss across K x HCA at best revert ----
grid = {}
for Kv in (12, 20, 28, 36):
    for Hv in (50, 75, 100, 125):
        pv, _ = run_elo(df, Kv, Hv, REVERT)
        grid[f"K{Kv}_H{Hv}"] = float(ll_vec(np.clip(pv[vm],1e-12,1-1e-12), y[vm].astype(float)).mean())
res["sensitivity_val_logloss"] = grid

# home advantage over eras (descriptive)
res["home_win_rate_by_decade"] = {}
for d0 in range(1950, 2020, 10):
    sel = (df.year_id >= d0) & (df.year_id < d0+10) & (df.hf == 1)
    if sel.sum() > 0:
        res["home_win_rate_by_decade"][f"{d0}s"] = float(df.loc[sel, "y"].mean())

print(json.dumps(res, indent=2))
