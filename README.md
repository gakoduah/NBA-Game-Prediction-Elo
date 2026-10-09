# Quantitative NBA Game Forecasting

A from-scratch **Elo rating model** that predicts NBA game outcomes, built and
evaluated with a strict **walk-forward, leakage-free** methodology on **real
historical data** and benchmarked against FiveThirtyEight's professional model.

---

## What this project does

Given every NBA/BAA game from 1947 to 2015, the system answers a concrete
predictive question: *how well can a simple, transparent rating model forecast
the winner of a game using only information available beforehand?*

It builds team ratings from first principles, processes games in strict
chronological order, predicts each game **before** seeing its result, and then
updates the ratings. Hyper-parameters are tuned on an early validation window,
**frozen**, and evaluated on a later held-out window that is never used for
tuning — so the reported numbers reflect genuine out-of-sample performance, not
a fit to the test data.

## Headline results (held-out test: 2010–2015, 7,641 real games)

| Model | Accuracy | Brier | Log-loss |
|---|---|---|---|
| Always pick home team | 59.5% | — | — |
| Base-rate (home win %) | 59.5% | 0.242 | 0.677 |
| **Elo (this work)** | **67.2%** | **0.208** | **0.603** |
| FiveThirtyEight forecast (benchmark) | 67.3% | 0.206 | 0.598 |

**Takeaway:** a from-scratch Elo model recovers essentially all of a
professional model's skill — within 0.1 points of accuracy and ~0.8% of
log-loss of FiveThirtyEight — while beating naive baselines by ~7.7 points. The
model is also well-calibrated: predicted probabilities match observed win
frequencies closely across the full range (see `fig1_calibration.png`).

Tuned parameters (selected on 1980–2009 only): `K = 20`, home-court advantage
`= 100` Elo points, between-season mean reversion `= 0.25`.

## Why the methodology is sound

* **No look-ahead bias.** Every prediction uses only ratings formed from prior
  games; ratings update only after the outcome is observed.
* **No tuning on the test set.** Hyper-parameters are chosen on 1980–2009 and
  frozen before touching 2010–2015.
* **Honest baselines.** Performance is reported against trivial baselines and
  against a published professional forecast, not against a straw man.
* **Real data.** Outcomes are actual NBA results, not simulated from a model.

## Data

FiveThirtyEight's complete NBA Elo dataset (`nbaallelo.csv`, ~18 MB, one row
per team per game). The script auto-downloads it on first run, or accepts a
local path. Source:
`https://github.com/fivethirtyeight/data/tree/master/nba-elo`

## Architecture

```
nba_game_prediction.py
├── load_data        — parse, de-duplicate to one row per game, sort chronologically
├── run_elo          — from-scratch walk-forward Elo (home court + season reversion)
├── metrics          — accuracy, Brier score, log loss
├── tuning loop      — grid search on the validation window only
├── evaluation       — frozen params on the held-out test window + baselines + 538
└── figures          — calibration curve, benchmark bars, accuracy by season
```

All models are implemented from first principles using only **NumPy**,
**pandas**, and **Matplotlib**.

## Installation & usage

```bash
pip install numpy pandas matplotlib
python3 nba_game_prediction.py            # auto-downloads the dataset
python3 nba_game_prediction.py data.csv   # or point at a local copy
```

Outputs are written to `./project_outputs/`:

```
project_outputs/
├── fig1_calibration.png      — predicted vs observed win frequency
├── fig2_benchmarks.png       — accuracy vs baselines and the 538 benchmark
├── fig3_season_accuracy.png  — test-period accuracy by season
└── summary.json              — all metrics and tuned parameters
```

Results are deterministic: the pipeline is a fixed sequence over sorted games
with no random component, so re-running reproduces the figures and numbers
exactly.

## Mathematics

**Elo expected score** (team rating `R_t`, opponent `R_o`, home field `h ∈ {−1,0,1}`,
home-court advantage `H`):

```
E_t = 1 / (1 + 10^((R_o − R_t − H·h) / 400))
```

**Update** after observing outcome `y ∈ {0,1}` with learning rate `K`:

```
R_t ← R_t + K·(y − E_t)
```

**Between-season mean reversion** (factor `r`, league mean `μ = 1500`):

```
R ← μ + (1 − r)·(R − μ)
```

**Scoring.** Brier score `= mean((p − y)²)`; log loss
`= −mean(y·log p + (1 − y)·log(1 − p))`.

## Scope and honest limitations

* This is an **outcome-forecasting** study. It makes **no betting or
  market-efficiency claims**: reliable historical closing odds were not
  available, so there is no wagering backtest here. Forecasting a winner and
  beating a vig-laden market are different problems, and only the first is
  claimed.
* Elo uses only rating state. Richer features (rest, travel, injuries,
  player-level data) and models (logistic regression, gradient boosting) are
  natural extensions and a direction for future work.

## License

MIT

## References

- Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present.* Arco.
- Silver, N. et al. FiveThirtyEight NBA Elo methodology and dataset.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of
  probability. *Monthly Weather Review*, 78(1), 1–3.
