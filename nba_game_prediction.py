"""
Quantitative NBA Game Forecasting
=================================
A from-scratch Elo rating model for predicting NBA game outcomes, built and
evaluated with a strict walk-forward (leakage-free) methodology on real data.

Data : FiveThirtyEight `nbaallelo.csv` (every NBA/BAA game, 1947-2015).
       Auto-downloaded on first run if not present locally.

Method
------
* One canonical row per game, processed in strict chronological order.
* Each game is predicted using ONLY ratings formed from prior games
  (no look-ahead), then the result updates the ratings.
* Hyper-parameters (K, home-court advantage, between-season mean reversion)
  are tuned ONLY on a validation window (1980-2009), then FROZEN and
  evaluated on a held-out test window (2010-2015) that is never used for tuning.
* Reported against three honest baselines, including FiveThirtyEight's own
  published pre-game forecast.

Dependencies: numpy, pandas, matplotlib.
Run:  python3 nba_game_prediction.py            # auto-downloads data
      python3 nba_game_prediction.py data.csv   # use a local copy
"""
from __future__ import annotations
import os
import sys
import json
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_URL = ("https://raw.githubusercontent.com/fivethirtyeight/data/"
            "master/nba-elo/nbaallelo.csv")
OUT = "project_outputs"

WARMUP_END = 1980          # ratings burn-in; games before this update but are not scored
VAL_WINDOW = (1980, 2009)  # hyper-parameter tuning window
TEST_WINDOW = (2010, 2015) # held-out evaluation window (never used for tuning)

K_GRID = (12, 20, 28, 36)
HCA_GRID = (50, 75, 100, 125)
REVERT_GRID = (0.0, 0.25, 0.5)


# ----------------------------------------------------------------------
#  Data
# ----------------------------------------------------------------------
def load_data(path: str | None) -> pd.DataFrame:
    if path is None:
        path = "nbaallelo.csv"
        if not os.path.exists(path):
            print(f"Downloading dataset -> {path} ...")
            urllib.request.urlretrieve(DATA_URL, path)
    df = pd.read_csv(path)
    df = df[df["_iscopy"] == 0].copy()               # one row per game
    df["date"] = pd.to_datetime(df["date_game"], format="%m/%d/%Y")
    df = df.sort_values(["date", "gameorder"]).reset_index(drop=True)
    df["y"] = (df["game_result"] == "W").astype(int)  # canonical-team win
    df["hf"] = df["game_location"].map({"H": 1, "A": -1, "N": 0}).fillna(0).astype(int)
    return df


# ----------------------------------------------------------------------
#  Elo (from first principles), walk-forward
# ----------------------------------------------------------------------
def run_elo(df: pd.DataFrame, K: float, hca: float, revert: float,
            base: float = 1500.0):
    """Return pre-game predicted win probabilities, outcomes, and season tags.

    Every prediction uses only information available before the game; ratings
    are updated only after the outcome is revealed, so the series is free of
    look-ahead bias by construction.
    """
    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}
    n = len(df)
    preds = np.empty(n)
    team = df["fran_id"].values
    opp = df["opp_fran"].values
    hf = df["hf"].values
    y = df["y"].values
    yr = df["year_id"].values

    for i in range(n):
        t, o = team[i], opp[i]
        rt = ratings.get(t, base)
        ro = ratings.get(o, base)
        if revert > 0:                                # mean-revert between seasons
            if last_season.get(t, yr[i]) != yr[i]:
                rt = base + (1 - revert) * (rt - base)
            if last_season.get(o, yr[i]) != yr[i]:
                ro = base + (1 - revert) * (ro - base)
        last_season[t] = yr[i]
        last_season[o] = yr[i]

        exp_t = 1.0 / (1.0 + 10 ** ((ro - rt - hca * hf[i]) / 400.0))
        preds[i] = exp_t                              # prediction BEFORE update

        ratings[t] = rt + K * (y[i] - exp_t)
        ratings[o] = ro + K * ((1 - y[i]) - (1 - exp_t))
    return preds, y, yr


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return {
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
    }


# ----------------------------------------------------------------------
#  Figures
# ----------------------------------------------------------------------
def fig_calibration(p, y, path):
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p, bins) - 1
    xs, ys, ns = [], [], []
    for b in range(10):
        sel = idx == b
        if sel.sum() > 0:
            xs.append(p[sel].mean()); ys.append(y[sel].mean()); ns.append(sel.sum())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(xs, ys, "o-", color="#1f77b4", label="Model")
    for x, yv, nb in zip(xs, ys, ns):
        ax.annotate(str(nb), (x, yv), fontsize=7, xytext=(3, -9),
                    textcoords="offset points", color="#555")
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed win frequency")
    ax.set_title("Calibration on held-out test (2010-2015)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_benchmarks(summary, path):
    labels = ["Always\nhome", "Base\nrate", "Elo\n(this work)", "538\nforecast"]
    acc = [summary["baseline_always_home"]["accuracy"],
           summary["baseline_base_rate"]["accuracy"],
           summary["elo_from_scratch"]["accuracy"],
           summary["fivethirtyeight_benchmark"]["accuracy"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#bbbbbb", "#bbbbbb", "#1f77b4", "#ff7f0e"]
    bars = ax.bar(labels, [a * 100 for a in acc], color=colors)
    for b, a in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, a * 100 + 0.4,
                f"{a*100:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(55, 70)
    ax.set_title("Out-of-sample accuracy vs baselines and professional benchmark")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_season_accuracy(df, p, y, yr, test_window, path):
    mask = (yr >= test_window[0]) & (yr <= test_window[1])
    seasons = sorted(set(yr[mask]))
    accs = []
    for s in seasons:
        sel = (yr == s)
        accs.append(np.mean((p[sel] >= 0.5) == (y[sel] == 1)) * 100)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(seasons, accs, "o-", color="#1f77b4")
    ax.axhline(np.mean(accs), ls="--", color="gray",
               label=f"Mean {np.mean(accs):.1f}%")
    ax.set_xlabel("Season"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Test-period accuracy by season")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_data(path)
    print(f"Games (canonical): {len(df):,}  seasons "
          f"{df.year_id.min()}-{df.year_id.max()}")

    # ---- Tune on validation window only ----
    best = None
    for K in K_GRID:
        for hca in HCA_GRID:
            for revert in REVERT_GRID:
                p, y, yr = run_elo(df, K, hca, revert)
                m = (yr >= VAL_WINDOW[0]) & (yr <= VAL_WINDOW[1])
                ll = metrics(p[m], y[m])["log_loss"]
                if best is None or ll < best[0]:
                    best = (ll, K, hca, revert)
    _, K, hca, revert = best
    print(f"Tuned on {VAL_WINDOW} -> K={K}, HCA={hca}, revert={revert}")

    # ---- Freeze params, evaluate on held-out test ----
    p, y, yr = run_elo(df, K, hca, revert)
    tm = (yr >= TEST_WINDOW[0]) & (yr <= TEST_WINDOW[1])
    elo_m = metrics(p[tm], y[tm])

    hf_t = df["hf"].values[tm]
    yt = y[tm]
    p_home = np.where(hf_t == 1, 1.0, np.where(hf_t == -1, 0.0, 0.5))
    base_rate = df.loc[(df.year_id >= VAL_WINDOW[0]) & (df.year_id <= VAL_WINDOW[1])
                       & (df.hf == 1), "y"].mean()
    p_base = np.where(hf_t == 1, base_rate, np.where(hf_t == -1, 1 - base_rate, 0.5))
    f538 = df["forecast"].values[tm].astype(float)
    v = ~np.isnan(f538)

    summary = {
        "n_games_total": int(len(df)),
        "season_range": f"{int(df.year_id.min())}-{int(df.year_id.max())}",
        "tuned_params": {"K": K, "home_court_adv": hca, "season_revert": revert},
        "validation_window": list(VAL_WINDOW),
        "test_window": list(TEST_WINDOW),
        "n_test_games": int(tm.sum()),
        "elo_from_scratch": elo_m,
        "baseline_always_home": {"accuracy": float(np.mean((p_home >= 0.5) == (yt == 1)))},
        "baseline_base_rate": metrics(p_base, yt),
        "fivethirtyeight_benchmark": metrics(f538[v], yt[v]),
    }

    fig_calibration(p[tm], yt, f"{OUT}/fig1_calibration.png")
    fig_benchmarks(summary, f"{OUT}/fig2_benchmarks.png")
    fig_season_accuracy(df, p, y, yr, TEST_WINDOW, f"{OUT}/fig3_season_accuracy.png")

    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== HELD-OUT TEST", TEST_WINDOW, f"(n={int(tm.sum()):,}) ===")
    print(f"  Elo (from scratch): acc={elo_m['accuracy']:.4f}  "
          f"Brier={elo_m['brier']:.4f}  logloss={elo_m['log_loss']:.4f}")
    print(f"  Always-home       : acc={summary['baseline_always_home']['accuracy']:.4f}")
    print(f"  Base-rate         : acc={summary['baseline_base_rate']['accuracy']:.4f}  "
          f"Brier={summary['baseline_base_rate']['brier']:.4f}  "
          f"logloss={summary['baseline_base_rate']['log_loss']:.4f}")
    print(f"  538 benchmark     : acc={summary['fivethirtyeight_benchmark']['accuracy']:.4f}  "
          f"Brier={summary['fivethirtyeight_benchmark']['brier']:.4f}  "
          f"logloss={summary['fivethirtyeight_benchmark']['log_loss']:.4f}")
    print(f"\nFigures and summary.json written to ./{OUT}/")


if __name__ == "__main__":
    main()
