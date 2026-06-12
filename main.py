"""Command-line orchestrator for the OFI pipeline.

Examples
--------
    python main.py ingest
    python main.py train
    python main.py evaluate
    python main.py all
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


def main() -> None:
    parser = argparse.ArgumentParser(description="OFI HFT pipeline")
    parser.add_argument(
        "stage",
        choices=["download", "ingest", "train", "evaluate", "all"],
    )
    parser.add_argument("--config", default="config.ini", type=Path)
    args = parser.parse_args()

    cfg = PipelineConfig.load(args.config)
    _setup_logging(cfg)

    if args.stage in {"download", "all"}:
        cmd_download(cfg)
    if args.stage in {"ingest", "all"}:
        cmd_ingest(cfg)
    if args.stage in {"train", "all"}:
        cmd_train(cfg)
    if args.stage in {"evaluate", "all"}:
        cmd_evaluate(cfg)


if __name__ == "__main__":
    main()
