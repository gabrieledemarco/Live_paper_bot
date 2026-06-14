"""Command-line orchestrator for the OFI pipeline.

Examples
--------
    python main.py ingest
    python main.py train
    python main.py evaluate
    python main.py all

High-Timeframe (HTF) multi-timeframe track:
    python main.py htf-download    # CCXT OHLCV ingestion (incremental)
    python main.py htf-features    # build + cache feature matrices (sanity check)
    python main.py htf-train       # purged-CV training, one model per pair
    python main.py htf-evaluate    # holdout metrics + reports + score table
    python main.py htf-all         # download -> train -> evaluate
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.core.config_loader import PipelineConfig
from src.core.data_manager import DataManager
from src.core.downloader import BinanceVisionDownloader
from src.models.evaluation import ModelEvaluator
from src.models.trainer import ModelTrainer


def _setup_logging(cfg: PipelineConfig) -> None:
    cfg.report.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(cfg.report.log_dir / "pipeline.log"),
        ],
    )


def cmd_download(cfg: PipelineConfig) -> None:
    dl = BinanceVisionDownloader(cfg.data.input_dir, market=cfg.data.market)
    # Cover both train and test windows in a single pass.
    start = min(cfg.data.train_start_date, cfg.data.test_start_date)
    end = max(cfg.data.train_end_date, cfg.data.test_end_date)
    dl.download_many(cfg.data.pairs, start, end, kinds=("bookTicker", "trades"))


def cmd_ingest(cfg: PipelineConfig) -> None:
    start = min(cfg.data.train_start_date, cfg.data.test_start_date)
    end = max(cfg.data.train_end_date, cfg.data.test_end_date)
    for pair in cfg.data.pairs:
        logging.info("Ingesting %s", pair)
        dm = DataManager(pair=pair,
                         input_dir=cfg.data.input_dir,
                         output_dir=cfg.data.output_dir,
                         market=cfg.data.market,
                         auto_download=cfg.data.auto_download,
                         download_range=(start, end))
        dm.persist()


def cmd_train(cfg: PipelineConfig) -> None:
    trainer = ModelTrainer(cfg)
    trainer.train_all()


def cmd_evaluate(cfg: PipelineConfig) -> None:
    evaluator = ModelEvaluator(cfg)
    evaluator.evaluate_all()


# --------------------------------------------------------------------------- #
# HTF multi-timeframe stages
# --------------------------------------------------------------------------- #
def _require_htf(cfg: PipelineConfig) -> None:
    if cfg.htf_data is None:
        raise SystemExit("HTF sections missing from config.ini ([HTF_DATA]/[HTF_MODEL]).")


def cmd_htf_download(cfg: PipelineConfig) -> None:
    _require_htf(cfg)
    from src.core.ccxt_downloader import CCXTOHLCVDownloader
    dl = CCXTOHLCVDownloader(output_dir=cfg.htf_data.output_dir,
                             exchange=cfg.htf_data.exchange)
    dl.download(cfg.htf_data.pairs, cfg.htf_data.timeframes, cfg.htf_data.lookback_days)


def cmd_htf_features(cfg: PipelineConfig) -> None:
    _require_htf(cfg)
    from src.models.htf_trainer import HTFTrainer
    trainer = HTFTrainer(cfg)
    for pair in cfg.htf_data.pairs:
        X_df, y, cols = trainer.build_matrix(pair)
        logging.info("[%s] features: %d rows x %d cols; label balance=%s",
                     pair, X_df.shape[0], X_df.shape[1],
                     dict(y.value_counts()) if cfg.htf_model.task_type == "classification"
                     else f"mean={y.mean():.2e}")


def cmd_htf_train(cfg: PipelineConfig) -> None:
    _require_htf(cfg)
    from src.models.htf_trainer import HTFTrainer
    HTFTrainer(cfg).train_all()


def cmd_htf_evaluate(cfg: PipelineConfig) -> None:
    _require_htf(cfg)
    from src.models.htf_evaluation import HTFEvaluator
    HTFEvaluator(cfg).evaluate_all()


def cmd_htf_backtest(cfg: PipelineConfig) -> None:
    _require_htf(cfg)
    if cfg.htf_backtest is None:
        raise SystemExit("Missing [HTF_BACKTEST] section in config.ini.")
    from src.models.htf_backtest_runner import HTFBacktestRunner
    from src.report.htf_dashboard import build_dashboard
    runner = HTFBacktestRunner(cfg)
    results = runner.run_all()
    charts_root = cfg.report.charts_dir.parent / "htf" / "backtest"
    out = charts_root / "dashboard.html"
    build_dashboard(results, charts_root=charts_root, out_path=out)
    logging.info("Dashboard written -> %s", out)
    print(f"\nHTF backtest dashboard: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OFI HFT + HTF multi-timeframe pipeline")
    parser.add_argument(
        "stage",
        choices=["download", "ingest", "train", "evaluate", "all",
                 "htf-download", "htf-features", "htf-train", "htf-evaluate", "htf-all",
                 "htf-backtest"],
    )
    parser.add_argument("--config", default="config.ini", type=Path)
    args = parser.parse_args()

    cfg = PipelineConfig.load(args.config)
    _setup_logging(cfg)

    # --- tick-level OFI track ---
    if args.stage in {"download", "all"}:
        cmd_download(cfg)
    if args.stage in {"ingest", "all"}:
        cmd_ingest(cfg)
    if args.stage in {"train", "all"}:
        cmd_train(cfg)
    if args.stage in {"evaluate", "all"}:
        cmd_evaluate(cfg)

    # --- HTF multi-timeframe track ---
    if args.stage in {"htf-download", "htf-all"}:
        cmd_htf_download(cfg)
    if args.stage == "htf-features":
        cmd_htf_features(cfg)
    if args.stage in {"htf-train", "htf-all"}:
        cmd_htf_train(cfg)
    if args.stage in {"htf-evaluate", "htf-all"}:
        cmd_htf_evaluate(cfg)
    if args.stage == "htf-backtest":
        cmd_htf_backtest(cfg)


if __name__ == "__main__":
    main()
