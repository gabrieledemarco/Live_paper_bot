"""Streamlit front-end for the OFI HFT pipeline.

Run from the project root with:

    streamlit run src/ui/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make `src` importable when launched via `streamlit run`.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.engine import BacktestEngine  # noqa: E402
from src.core.config_loader import PipelineConfig  # noqa: E402
from src.core.data_manager import DataManager  # noqa: E402
from src.core.downloader import BinanceVisionDownloader  # noqa: E402
from src.core.features import OFIFeatureBuilder  # noqa: E402
from src.models.evaluation import ModelEvaluator  # noqa: E402
from src.models.trainer import ModelTrainer  # noqa: E402

st.set_page_config(page_title="OFI HFT Pipeline", layout="wide")

CONFIG_PATH = ROOT / "config.ini"


@st.cache_resource(show_spinner=False)
def _load_config() -> PipelineConfig:
    return PipelineConfig.load(CONFIG_PATH)


def _save_config(text: str) -> None:
    CONFIG_PATH.write_text(text)
    _load_config.clear()


cfg = _load_config()

st.title("OFI HFT Trading Pipeline")
st.caption("Order Flow Imbalance microstructure strategy - parametric end-to-end framework")

TAB_CFG, TAB_TRAIN, TAB_ANALYTICS, TAB_BT = st.tabs([
    "1) Configurazione & Ingestion",
    "2) Addestramento & Validazione",
    "3) Analisi Grafica Avanzata",
    "4) Live Backtest Simulator",
])

# ---------------------------------------------------------------------- #
# TAB 1 - Config & Ingestion
# ---------------------------------------------------------------------- #
with TAB_CFG:
    st.header("Configurazione (config.ini)")
    text = st.text_area("INI content", CONFIG_PATH.read_text(),
                        height=400, key="cfg_editor")
    col1, col2 = st.columns(2)
    if col1.button("Salva configurazione"):
        _save_config(text)
        st.success("config.ini aggiornato.")
        cfg = _load_config()

    st.divider()
    st.header("Ingestion dei dati di mercato")
    st.write("Decomprime gli archivi Binance Vision (bookTicker + trades) e li "
             "persiste come Parquet partizionato per coppia e data.")

    target = st.selectbox("Coppia", cfg.data.pairs, key="ingest_pair")

    dl_col, ing_col = st.columns(2)
    if dl_col.button("Scarica dati da Binance Vision"):
        progress = st.progress(0.0, text=f"Download {target} ...")
        try:
            dl = BinanceVisionDownloader(cfg.data.input_dir, market=cfg.data.market)
            start = min(cfg.data.train_start_date, cfg.data.test_start_date)
            end = max(cfg.data.train_end_date, cfg.data.test_end_date)
            reports = dl.download_pair(target, start, end,
                                       kinds=("bookTicker", "trades"))
            progress.progress(1.0, text="Download completato.")
            for r in reports:
                st.write(f"**{r.kind}** - scaricati: {len(r.downloaded)}, "
                         f"già presenti: {len(r.skipped)}, "
                         f"mancanti su Vision: {len(r.missing)}")
        except Exception as exc:
            st.error(f"Errore download: {exc}")

    if ing_col.button("Esegui ingestion"):
        progress = st.progress(0.0, text=f"Avvio ingestion {target} ...")
        try:
            start = min(cfg.data.train_start_date, cfg.data.test_start_date)
            end = max(cfg.data.train_end_date, cfg.data.test_end_date)
            dm = DataManager(pair=target,
                             input_dir=cfg.data.input_dir,
                             output_dir=cfg.data.output_dir,
                             market=cfg.data.market,
                             auto_download=cfg.data.auto_download,
                             download_range=(start, end))
            progress.progress(0.25, text="Lettura bookTicker ...")
            book = dm.load_book_ticker()
            progress.progress(0.5, text="Lettura trades ...")
            trades = dm.load_trades()
            progress.progress(0.7, text="Merge tick stream ...")
            merged = dm.build_tick_stream()
            progress.progress(0.85, text="Scrittura Parquet ...")
            out = dm.persist(merged)
            progress.progress(1.0, text=f"Completato -> {out}")
            st.success(f"Tick stream salvato in {out} ({len(merged):,} eventi).")
        except Exception as exc:
            st.error(f"Errore durante ingestion: {exc}")

# ---------------------------------------------------------------------- #
# TAB 2 - Training & Validation
# ---------------------------------------------------------------------- #
with TAB_TRAIN:
    st.header("Addestramento e validazione")
    st.write("Lancia il `ModelTrainer` per tutte le coppie configurate "
             "(Time Series CV + persistenza del modello).")

    if st.button("Esegui training per tutte le coppie"):
        with st.spinner("Training in corso ..."):
            trainer = ModelTrainer(cfg)
            results = trainer.train_all()
        for r in results:
            st.subheader(r.pair)
            st.json({
                "model_path": str(r.model_path),
                "cv_macro_f1": r.cv_scores,
                "feature_columns": r.feature_columns,
            })
            st.dataframe(pd.DataFrame(r.classification_report).T)

    st.divider()
    st.header("Heatmap confusion matrix")
    pair = st.selectbox("Coppia da ispezionare", cfg.data.pairs, key="train_pair")
    cm_path = cfg.report.charts_dir / pair / "confusion_matrix.png"
    summary_path = cfg.report.charts_dir / pair / "summary.json"
    if cm_path.exists():
        st.image(str(cm_path))
    else:
        st.info("Esegui prima l'evaluation per generare la confusion matrix.")

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        st.write("Classification report:")
        st.dataframe(pd.DataFrame(summary["classification_report"]).T)

# ---------------------------------------------------------------------- #
# TAB 3 - Advanced analytics
# ---------------------------------------------------------------------- #
with TAB_ANALYTICS:
    st.header("Analisi grafica avanzata")
    pair = st.selectbox("Coppia", cfg.data.pairs, key="ana_pair")

    if st.button("Genera evaluation completa"):
        with st.spinner("Calcolo metriche e grafici ..."):
            evaluator = ModelEvaluator(cfg)
            evaluator.evaluate_pair(pair)
        st.success("Report rigenerato.")

    charts_dir = cfg.report.charts_dir / pair
    sections = [
        ("Confusion Matrix", "confusion_matrix.png",
         "Distribuzione predizioni vs valori reali sulle classi {-1, 0, +1}."),
        ("Rolling Sharpe & Information Ratio", "rolling_finance.png",
         "Andamento cumulativo dei rapporti di rischio annualizzati."),
        ("Alpha Decay", "alpha_decay.png",
         "Degrado dell'accuratezza predittiva al crescere dell'orizzonte k."),
        ("Latency Stress Test", "latency_stress.png",
         "Sharpe e fill-rate al variare della latenza artificiale."),
        ("Inventory & Idle Time", "inventory.png",
         "Esposizione netta long/short e percentuale di tempo flat."),
    ]
    for title, fname, desc in sections:
        st.subheader(title)
        st.markdown(desc)
        path = charts_dir / fname
        if path.exists():
            st.image(str(path))
        else:
            st.info(f"`{fname}` non ancora generato.")

# ---------------------------------------------------------------------- #
# TAB 4 - Live backtest
# ---------------------------------------------------------------------- #
with TAB_BT:
    st.header("Live backtest simulator")
    col1, col2, col3 = st.columns(3)
    pair = col1.selectbox("Coppia", cfg.data.pairs, key="bt_pair")
    latency = col2.number_input("Latency (ms)", min_value=0,
                                value=int(cfg.backtest.base_latency_ms), step=5)
    capital = col3.number_input("Initial capital",
                                value=float(cfg.backtest.initial_capital), step=1000.0)

    if st.button("Esegui backtest"):
        with st.spinner("Simulazione event-driven in corso ..."):
            import joblib
            bundle_path = cfg.model.model_dir / f"{pair}_ofi_{cfg.model.model_type.lower()}.joblib"
            if not bundle_path.exists():
                st.error(f"Modello mancante: {bundle_path}. Esegui prima il training.")
            else:
                bundle = joblib.load(bundle_path)
                dm = DataManager(pair=pair,
                                 input_dir=cfg.data.input_dir,
                                 output_dir=cfg.data.output_dir)
                ticks = dm.load_partitioned(
                    start_date=cfg.data.test_start_date,
                    end_date=cfg.data.test_end_date,
                )
                builder = OFIFeatureBuilder(
                    resample_freq=cfg.data.resample_freq,
                    rolling_window=cfg.model.ofi_window,
                    target_ticks=cfg.model.target_ticks,
                    threshold_bps=cfg.model.threshold_alpha,
                )
                X_df, y = builder.build(ticks)

                cfg.backtest.initial_capital = float(capital)
                engine = BacktestEngine(cfg)
                res = engine.run(pair=pair, model_bundle=bundle, tick_stream=ticks,
                                 X_df=X_df, y=y, latency_ms=int(latency))

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Sharpe", f"{res['kpis']['sharpe']:.2f}")
                k2.metric("Max Drawdown", f"{res['kpis']['max_drawdown']:.2%}")
                k3.metric("Fill Rate", f"{res['fill_rate']:.2%}")
                k4.metric("Total Return", f"{res['kpis']['total_return']:.2%}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res["equity"].index,
                                         y=res["equity"].values,
                                         mode="lines", name="Equity"))
                fig.update_layout(title=f"{pair} - Equity Curve",
                                  xaxis_title="Time", yaxis_title="Equity")
                st.plotly_chart(fig, use_container_width=True)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=res["inventory"].index,
                                          y=res["inventory"].values,
                                          mode="lines", name="Inventory",
                                          line=dict(color="seagreen")))
                fig2.update_layout(title="Inventory over time",
                                   xaxis_title="Time", yaxis_title="Net position")
                st.plotly_chart(fig2, use_container_width=True)
