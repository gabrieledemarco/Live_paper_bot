# PDR — Project Design Report
## OFI HFT Trading Pipeline

| | |
|---|---|
| **Progetto** | OFI HFT Trading Pipeline |
| **Versione** | 1.0 |
| **Data** | 2026-06-12 |
| **Branch** | `claude/serene-cannon-7mhhmk` |
| **Owner** | Gabriele De Marco |
| **Stato** | Funzionante end-to-end (validato su dati sintetici + reali Binance Vision) |

---

## 1. Executive Summary

Framework OOP **parametrico e agnostico rispetto alla coppia** per una strategia di
trading ad alta frequenza basata sulla microstruttura del Limit Order Book, nello
specifico sull'**Order Flow Imbalance (OFI) di Livello 1**.

Tutto il comportamento (asset, date, soglie, latenze, fee, modello) è centralizzato in
un singolo file `config.ini`: la sostanza del metodo matematico e simulativo resta
invariata al variare degli input. La pipeline copre l'intero ciclo di vita:
ingestion tick-by-tick → feature engineering vettorializzato → addestramento con
validazione temporale → backtest event-driven → reportistica analitica → UI web.

---

## 2. Obiettivi e Non-Obiettivi

### 2.1 Obiettivi
- Pipeline riproducibile e parametrica per testare il potere predittivo dell'OFI.
- Storage tick-by-tick (mai OHLC/klines) partizionato per coppia/data.
- Backtester realistico: worst-case SL/TP, queue position teorica, latenza.
- Metriche di classificazione (scikit-learn) e finanziarie (empyrical/quantstats)
  senza reimplementare formule standard.
- Front-end web completo per gestire l'intero ciclo dal browser.

### 2.2 Non-Obiettivi (anti-overkill, per scelta)
- Nessun modello di Deep Learning: l'OFI lineare + modelli ad albero sono la baseline.
- Nessuna feature esotica oltre l'OFI L1 e i suoi derivati rolling.
- Nessun trading live / order routing reale (solo simulazione).
- Nessun book oltre il Livello 1 (Top-of-Book).

---

## 3. Stakeholder & Personas
- **Quant Researcher** — regola `config.ini`, valuta alpha decay e metriche.
- **Strategy Developer** — estende feature/modelli mantenendo l'interfaccia.
- **Operatore** — usa la UI Streamlit per scaricare dati, addestrare, simulare.

---

## 4. Architettura del Sistema

```
config.ini                  # configurazione centralizzata (unica fonte di verità)
main.py                     # CLI: download | ingest | train | evaluate | all
scripts/gen_synthetic.py    # generatore dati sintetici Binance-shaped (sandbox)
src/
├── core/
│   ├── config_loader.py    # PipelineConfig tipizzato; path ancorati alla root
│   ├── downloader.py       # BinanceVisionDownloader (spot/um/cm, parallelo)
│   ├── data_manager.py     # ingestion → tick stream → Parquet partizionato
│   └── features.py         # OFIFeatureBuilder vettorializzato (NumPy)
├── backtest/
│   └── engine.py           # BacktestEngine + LatencyStressTester
├── models/
│   ├── trainer.py          # ModelTrainer (TimeSeriesSplit, Ridge|LightGBM)
│   └── evaluation.py       # ModelEvaluator + financial_kpis
└── ui/
    └── app.py              # front-end Streamlit a 4 tab
data/raw/<PAIR>/{bookTicker,trades}/   # archivi grezzi Binance Vision
data/parquet/<PAIR>/date=YYYY-MM-DD/   # tick stream partizionato
models_store/<PAIR>_ofi_<model>.joblib # un modello per coppia
reports/charts/<PAIR>/                 # grafici + summary.json
```

### 4.1 Flusso dati end-to-end
```
Binance Vision (zip)
   └─[downloader]→ data/raw/<PAIR>/
        └─[data_manager]→ merge BBO+trades ordinato per ts → data/parquet/<PAIR>/
             └─[features]→ OFI L1 + rolling + label {-1,0,+1}
                  ├─[trainer]→ TimeSeriesSplit → models_store/<PAIR>.joblib
                  └─[engine]→ backtest event-driven → equity, KPI, inventory
                       └─[evaluation]→ reports/charts/<PAIR>/*.png + summary.json
                            └─[ui]→ rendering interattivo nel browser
```

---

## 5. Componenti — Specifiche Funzionali

### 5.1 `config_loader.py`
- Espone `PipelineConfig` (dataclass tipizzati: `DataCfg`, `ModelCfg`, `BacktestCfg`, `ReportCfg`).
- **Tutti i percorsi relativi sono ancorati alla cartella di `config.ini`** (la root del
  progetto), non alla working directory del processo → deterministico su Streamlit Cloud.

### 5.2 `downloader.py` — `BinanceVisionDownloader`
- Scarica gli archivi giornalieri `bookTicker` e `trades`/`aggTrades` da
  `data.binance.vision` in parallelo (ThreadPool), tollerante ai 404.
- Mercati: `spot` (no bookTicker), `um` (USD-M futures), `cm` (COIN-M futures).
- Restituisce un `DownloadReport` (downloaded / skipped / missing per giorno).

### 5.3 `data_manager.py` — `DataManager(pair, ...)`
- Decomprime e legge i CSV nativi Binance (con/senza header), normalizza i tipi.
- **Unisce BBO + trades in un unico tick stream cronologico al millisecondo** sul
  timestamp dell'exchange; forward-fill dello stato BBO sulle righe trade.
- **Divieto assoluto di aggregazione OHLC**: solo eventi tick-by-tick.
- Persiste in Parquet snappy partizionato per `date=`.
- **Auto-bootstrap**: se mancano i Parquet (e `auto_download=true`) scarica e
  ingerisce automaticamente; in caso di fallimento l'errore spiega la causa
  (404 di Vision vs irraggiungibilità di rete).
- Calcola `signed_qty` (+ buyer-taker / − seller-taker); default robusto se manca
  `is_buyer_maker`.

### 5.4 `features.py` — `OFIFeatureBuilder`
- OFI L1 canonico (Cont-Kukanov-Stoikov) **interamente vettorializzato in NumPy**
  (nessun loop Python per riga).
- Aggregazione flessibile: finestra temporale (`resample_freq`) o tick.
- Feature: `ofi`, `ofi_norm` (z-score rolling), somme/medie/std rolling, `trade_flow`,
  `spread_norm`, `mid_return`.
- **Colonne di contesto** (`mid_price`, `bid_qty`, `ask_qty`) trasportate per il
  backtest ma escluse dai regressori (`FEATURE_COLUMNS` vs `CONTEXT_COLUMNS`).
- Label: segno della variazione futura del mid-price a `target_ticks`, soglia in bps.
- Guard: `target_ticks > 0`, lunghezza minima, esclusione mid ≤ 0/NaN.

### 5.5 `trainer.py` — `ModelTrainer`
- Cicla su tutte le coppie del config; per ciascuna: build feature →
  **TimeSeriesSplit rigoroso** → refit finale → serializza bundle joblib
  (modello + feature columns + snapshot config).
- Modelli: `RidgeClassifier` o `LGBMClassifier` (multiclass, `class_weight=balanced`),
  entrambi dentro una `Pipeline` con `StandardScaler`.
- Metriche CV: macro-F1 e accuracy (scikit-learn).

### 5.6 `engine.py` — `BacktestEngine` (event-driven)
- **Worst-case SL/TP**: se nello stesso intervallo il range tocca sia stop sia take,
  si assume lo **stop preso per primo**.
- **Queue position teorica**: ordine limit passivo posizionato dietro al **volume reale
  displayed** (`best_bid_qty`/`best_ask_qty`) al momento dell'immissione; eseguito solo
  quando i trade reali successivi consumano la coda.
- **Pricing coerente**: entry/exit/mark sul prezzo del lato della posizione (bid per
  long, ask per short) → nessun profitto fittizio da mezzo spread.
- **Latenza**: shift del segnale in barre + slippage avverso ∝ latenza + penalità di
  crescita coda (degrada fill rate). Parametrico via `latency_slippage_bps_per_ms`.
- Fee maker (fill passivo) / taker (uscita aggressiva).
- Output: equity curve, inventory series, rendimenti per-trade, KPI, fill rate, idle %.
- `LatencyStressTester`: sweep della griglia di latenza → Sharpe / fill rate / max DD.

### 5.7 `evaluation.py` — `ModelEvaluator`
Genera in `reports/charts/<PAIR>/`:
- **Confusion matrix** (heatmap seaborn) + classification report (precision/recall/F1).
- **Rolling Sharpe & Information Ratio** (annualizzazione 365g crypto).
- **Alpha decay reale**: hit-rate direzionale del segnale fisso vs segno del forward
  return a k+1/k+2/k+5/k+10 (curva monotòna verso il random 0.5).
- **Latency stress test**: Sharpe e fill rate vs latenza (0/10/50/100 ms).
- **Inventory & idle time**: esposizione netta long/short e % tempo flat.
- `financial_kpis()`: delega a empyrical (fallback quantstats / NumPy corretto);
  Sharpe/Sortino **per-trade** (robusti agli idle bar), Max DD, Profit Factor finito.

### 5.8 `ui/app.py` — Streamlit (4 tab)
1. **Configurazione & Ingestion** — editor `config.ini`, download e ingest con progress,
   pannello "Stato dati" persistente (archivi raw / partizioni / modello per coppia).
2. **Addestramento & Validazione** — bottone **Setup completo** (download+ingest+train
   con `st.status` live), training, confusion matrix e classification report.
3. **Analisi Grafica Avanzata** — rendering di tutti i grafici con descrizioni Markdown.
4. **Live Backtest Simulator** — parametri, equity curve interattiva (Plotly), Max DD,
   Fill Rate, metriche per-trade, inventory.

---

## 6. Parametri di Configurazione (`config.ini`)

| Sezione | Chiave | Significato |
|---|---|---|
| `[DATA]` | `input_dir`, `output_dir` | cartelle raw / parquet (relative → ancorate alla root) |
| | `market` | `spot` \| `um` \| `cm` (bookTicker solo per futures) |
| | `auto_download` | scarica automaticamente se mancano gli archivi |
| | `pairs` | lista coppie da elaborare |
| | `train/test_start/end_date` | finestre temporali train/test |
| | `resample_freq` | frequenza di aggregazione OFI (es. `1s`, `100ms`) |
| `[MODEL]` | `target_ticks` | orizzonte k futuro per la label |
| | `threshold_alpha` | soglia in bps per le classi {-1,0,+1} |
| | `model_type` | `Ridge` \| `LightGBM` |
| | `n_splits`, `ofi_window` | fold CV / finestra rolling OFI |
| `[BACKTEST]` | `initial_capital`, `maker_fee`, `taker_fee` | capitale e commissioni |
| | `base_latency_ms`, `latency_grid` | latenza base e griglia stress test |
| | `latency_slippage_bps_per_ms` | slippage avverso per ms di latenza |
| | `stop_loss_bps`, `take_profit_bps`, `max_position`, `signal_threshold` | regole di rischio |
| `[REPORT]` | `charts_dir`, `log_dir` | output grafici / log |

---

## 7. Metriche & Criteri di Accettazione

| Categoria | Metriche |
|---|---|
| Classificazione | Accuracy, Precision/Recall/F1 per classe, macro-F1, confusion matrix |
| Segnale | Alpha decay (hit-rate direzionale a k crescenti) |
| Finanziarie | Sharpe & Sortino per-trade, Max Drawdown, Profit Factor, Win Rate |
| Esecuzione | Fill Rate (condizionato alla coda), Idle Time %, Latency degradation |

**Criterio di correttezza tecnica** (validato): la pipeline gira end-to-end senza
errori, `ruff`/`pyflakes` puliti, e su dati sintetici senza vero alpha produce Sharpe
negativo (comportamento corretto: si pagano spread + fee).

---

## 8. Stack Tecnologico
- **Python 3.11+**
- numpy, pandas, pyarrow — dati e tick stream
- scikit-learn, lightgbm — modelli e metriche di classificazione
- empyrical-reloaded (+ pytz), quantstats — metriche finanziarie
- matplotlib, seaborn, plotly — grafici statici e interattivi
- streamlit — front-end web
- joblib — serializzazione modelli

---

## 9. Deployment & Esecuzione

```bash
pip install -r requirements.txt
python main.py download    # scarica archivi grezzi
python main.py ingest      # crea il tick stream Parquet
python main.py train       # addestra un modello per coppia
python main.py evaluate    # genera report e grafici
python main.py all         # esegue tutta la catena
streamlit run src/ui/app.py
```

**Streamlit Cloud**: il filesystem è effimero (azzerato ad ogni restart). Usare il
bottone **Setup completo** del Tab 2 per riscaricare + ingerire + addestrare in un
unico passaggio con feedback live.

---

## 10. Limitazioni Note & Rischi

| # | Limitazione | Mitigazione / Nota |
|---|---|---|
| 1 | Binance Vision non pubblica `bookTicker` per lo spot | usare `market=um/cm` o un provider con BBO spot (Tardis/Kaiko) |
| 2 | Filesystem Streamlit Cloud effimero | Setup completo riscarica on-demand |
| 3 | Solo Top-of-Book (L1) | sufficiente per OFI L1; L2/L3 fuori scope |
| 4 | Range intrabar approssimato col mid bar-over-bar | proxy worst-case ragionevole a 1s |
| 5 | Slippage da latenza è un modello lineare semplice | parametrico, calibrabile su dati reali |
| 6 | Reachability di `data.binance.vision` dipende dall'ambiente | errore diagnostico esplicito |

---

## 11. Roadmap / Estensioni Future
- Adapter multi-provider (Tardis.dev, Databento, LOBSTER) dietro la stessa interfaccia.
- Test suite `pytest` per congelare i comportamenti (no look-ahead, latenza monotòna,
  queue su volume reale).
- OFI multi-livello (L2) e feature di profondità.
- Walk-forward retraining e position sizing dinamico.
- Persistenza dei dati su storage esterno (S3) per superare l'effimero del cloud.

---

## 12. Changelog Sintetico
- **v1.0** — architettura completa: ingestion, OFI, training, backtest event-driven,
  evaluation, UI 4 tab. Downloader Binance Vision + auto-bootstrap. Fix di correttezza
  (pricing coerente, queue su volume reale, latenza efficace, metriche per-trade,
  alpha decay reale, path ancorati alla root, UX errori su Streamlit Cloud).
