# OFI HFT Trading Pipeline

End-to-end, fully parametric OOP framework for an HFT strategy based on
Level-1 **Order Flow Imbalance** (OFI). The pipeline is agnostic to the
underlying trading pair: every input (assets, dates, thresholds, latency)
is centralised in `config.ini` while the mathematical and simulation logic
is kept invariant.

## Project layout

```
.
├── config.ini                  # central parametric configuration
├── main.py                     # CLI orchestrator (ingest / train / evaluate / all)
├── requirements.txt
├── data/
│   ├── raw/                    # drop Binance Vision zip / csv archives here
│   └── parquet/<PAIR>/date=.../part.parquet
├── models_store/               # serialized models (one per pair)
├── reports/charts/<PAIR>/      # evaluation artefacts
└── src/
    ├── core/
    │   ├── config_loader.py    # typed PipelineConfig
    │   ├── data_manager.py     # ingestion + tick-by-tick parquet store
    │   └── features.py         # vectorised OFI feature builder
    ├── backtest/
    │   └── engine.py           # event-driven backtester + latency stressor
    ├── models/
    │   ├── trainer.py          # TimeSeriesSplit training loop
    │   └── evaluation.py       # classification + financial reporting
    └── ui/
        └── app.py              # Streamlit front-end (4 tabs)
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. drop Binance Vision archives in data/raw/
# 2. ingest tick-by-tick data into parquet store
python main.py ingest
# 3. train one model per pair
python main.py train
# 4. produce evaluation charts + backtest KPIs
python main.py evaluate
# 5. interactive UI
streamlit run src/ui/app.py
```
