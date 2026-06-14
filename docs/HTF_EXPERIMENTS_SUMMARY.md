# HTF Predictive Modeling — Experiments Summary

**Date:** 2026-06-14
**Data:** BTCUSDT + ETHUSDT · timeframes 1m/5m/15m/60m · ~180 days (CCXT/Binance)
**Base / horizon:** 1m base, forward horizon h=5 min (unless stated)
**Validation:** time-ordered holdout (last 20%) with embargo; purged walk-forward CV.

> Reproduce: `scripts/compare_models.py`, `compare_arima.py`, `horizon_sweep.py`,
> `exp_vpin.py`, `exp_meta.py`, `exp_sequence.py`. CSVs under
> `reports/htf/_comparison/`.

---

## 1. Model comparison (classification, h=5, fixed-threshold label)

| Model | BTC F1 | BTC AUC | ETH F1 | ETH AUC | Mean F1 |
|-------|-------:|--------:|-------:|--------:|--------:|
| LightGBM | 0.445 | 0.646 | 0.430 | 0.625 | **0.4376** |
| RandomForest | 0.446 | 0.656 | 0.428 | 0.635 | 0.4372 |
| HistGB | 0.442 | 0.639 | 0.427 | 0.616 | 0.4344 |
| Logistic | 0.434 | 0.657 | 0.410 | 0.634 | 0.4216 |
| Ridge | 0.427 | — | 0.399 | — | 0.4130 |

LightGBM ≈ RandomForest (within noise). Tree ensembles lead; Logistic is a
strong, fast linear baseline (best BTC AUC). Ridge has no probabilities.

## 2. ML vs ARIMA / SARIMA (regression, 60m, h=5 bars, IC)

| Model | Mean IC |
|-------|--------:|
| Ridge | −0.006 |
| ARIMA(2,0,2) | −0.009 |
| LightGBM | −0.014 |
| SARIMA(2,0,2)(1,0,1,24) | −0.019 |
| RandomForest | −0.027 |

All ~0, R²<0 everywhere: no model beats the naive mean at 60m/5h. SARIMA adds
nothing over ARIMA at ~8× the cost. (50-origin smoke test gave inflated IC up to
0.27 — small-sample noise; full ~864-origin walk-forward collapses to zero.)

## 3. Horizon sweep (1m, tercile labels, LightGBM)

| Horizon | BTC AUC | ETH AUC |
|--------:|--------:|--------:|
| 1 min | **0.636** | **0.633** |
| 3 min | 0.623 | 0.621 |
| 5 min | 0.622 | 0.616 |
| 10 min | 0.615 | 0.607 |
| 15 min | 0.606 | 0.601 |
| 30 min | 0.600 | 0.595 |
| 60 min | 0.590 | 0.584 |

Monotonic decay (matches Lucchese, Pakkanen & Veraart 2024). Working horizon
chosen: **h=5 min** (good AUC, more tradable than h=1).

## 4. VPIN + volume-clock features (h=5)

| Pair | Baseline AUC | +VPIN AUC | ΔF1 |
|------|-------------:|----------:|----:|
| BTC | 0.6203 | 0.6223 | +0.0021 |
| ETH | 0.6160 | 0.6155 | −0.0005 |

Negligible: VPIN/BVC/dollar-volume already captured by existing multi-TF proxies.

## 5. Triple-barrier + meta-labeling (h=5, PT=SL=1·σ, meta_thr=0.55)

| Pair | Primary precision (cov) | Meta precision (cov) | Gain |
|------|------------------------:|---------------------:|-----:|
| BTC | 0.473 (0.80) | 0.485 (0.19) | +0.012 |
| ETH | 0.480 (0.81) | 0.491 (0.21) | +0.012 |

Meta-labeling = selectivity lever: trades ~1/4 as often at higher precision.

## 6. Temporal-window neural vs GBT (h=5, L=8)

| Model | BTC AUC | ETH AUC |
|-------|--------:|--------:|
| LightGBM (window) | **0.624** | **0.620** |
| LightGBM (point-in-time) | 0.620 | 0.616 |
| MLP (window) | 0.577 | 0.577 |

Temporal context gives GBT +0.004; the shallow MLP loses clearly. Gradient
boosting confirmed as the right model family for this tabular feature set.

---

## Conclusion & recommendation

**Model choice is no longer the bottleneck** — everything clusters at AUC ~0.62 /
F1 ~0.43 at h=5; feature additions are marginal/null; the MLP underperforms. The
decisive question is now **economic**: does AUC ~0.62 survive ~4 bps taker fees,
slippage and latency? Classification metrics cannot answer this.

**Next step:** build a **cost-aware HTF bar-level backtester** (the existing
tick/OFI `backtest/engine.py` is not reusable) and run the top-3 configs:

1. **LightGBM on temporal window** (1m, h=5) — best AUC, robust.
2. **LightGBM + triple-barrier + meta-labeling** — precision-first; likely best
   net of costs because it trades selectively.
3. **RandomForest** (1m, h=5) — decorrelated second model, ensemble/robustness.

(MLP excluded — loses to GBT. Logistic optional fast linear floor.)

**Caveat:** ~4 bps taker fee vs ~5 bps move threshold ⇒ naive edge is razor-thin;
viability depends on selectivity (meta-labeling) and position sizing.

*Note: literature references in the chat synthesis were given from model knowledge
because WebSearch/WebFetch were unavailable this session (backend error); re-verify
live before citing.*
