"""Centralised configuration loader.

Reads `config.ini` and exposes a typed `PipelineConfig` dataclass that is
passed across all modules. Having a single place that interprets the INI
file keeps the rest of the codebase parameter-agnostic.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DataCfg:
    input_dir: Path
    output_dir: Path
    pairs: List[str]
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    resample_freq: str
    market: str = "spot"
    auto_download: bool = True


@dataclass
class ModelCfg:
    target_ticks: int
    threshold_alpha: float
    model_type: str
    n_splits: int
    ofi_window: int
    model_dir: Path


@dataclass
class BacktestCfg:
    initial_capital: float
    maker_fee: float
    taker_fee: float
    base_latency_ms: int
    latency_grid: List[int]
    stop_loss_bps: float
    take_profit_bps: float
    max_position: float
    signal_threshold: float
    latency_slippage_bps_per_ms: float = 0.02


@dataclass
class ReportCfg:
    charts_dir: Path
    log_dir: Path


@dataclass
class PipelineConfig:
    data: DataCfg
    model: ModelCfg
    backtest: BacktestCfg
    report: ReportCfg
    raw: configparser.ConfigParser = field(repr=False)

    @classmethod
    def load(cls, path: str | Path = "config.ini") -> "PipelineConfig":
        cfg = configparser.ConfigParser()
        read = cfg.read(path)
        if not read:
            raise FileNotFoundError(f"Config file not found: {path}")

        data = DataCfg(
            input_dir=Path(cfg.get("DATA", "input_dir")),
            output_dir=Path(cfg.get("DATA", "output_dir")),
            pairs=[p.strip().upper() for p in cfg.get("DATA", "pairs").split(",")],
            train_start_date=cfg.get("DATA", "train_start_date"),
            train_end_date=cfg.get("DATA", "train_end_date"),
            test_start_date=cfg.get("DATA", "test_start_date"),
            test_end_date=cfg.get("DATA", "test_end_date"),
            resample_freq=cfg.get("DATA", "resample_freq"),
            market=cfg.get("DATA", "market", fallback="spot"),
            auto_download=cfg.getboolean("DATA", "auto_download", fallback=True),
        )

        model = ModelCfg(
            target_ticks=cfg.getint("MODEL", "target_ticks"),
            threshold_alpha=cfg.getfloat("MODEL", "threshold_alpha"),
            model_type=cfg.get("MODEL", "model_type"),
            n_splits=cfg.getint("MODEL", "n_splits"),
            ofi_window=cfg.getint("MODEL", "ofi_window"),
            model_dir=Path(cfg.get("MODEL", "model_dir")),
        )

        backtest = BacktestCfg(
            initial_capital=cfg.getfloat("BACKTEST", "initial_capital"),
            maker_fee=cfg.getfloat("BACKTEST", "maker_fee"),
            taker_fee=cfg.getfloat("BACKTEST", "taker_fee"),
            base_latency_ms=cfg.getint("BACKTEST", "base_latency_ms"),
            latency_grid=[int(x.strip()) for x in cfg.get("BACKTEST", "latency_grid").split(",")],
            stop_loss_bps=cfg.getfloat("BACKTEST", "stop_loss_bps"),
            take_profit_bps=cfg.getfloat("BACKTEST", "take_profit_bps"),
            max_position=cfg.getfloat("BACKTEST", "max_position"),
            signal_threshold=cfg.getfloat("BACKTEST", "signal_threshold"),
            latency_slippage_bps_per_ms=cfg.getfloat(
                "BACKTEST", "latency_slippage_bps_per_ms", fallback=0.02),
        )

        report = ReportCfg(
            charts_dir=Path(cfg.get("REPORT", "charts_dir")),
            log_dir=Path(cfg.get("REPORT", "log_dir")),
        )

        # Ensure mandatory output dirs exist.
        for p in (data.output_dir, model.model_dir, report.charts_dir, report.log_dir):
            p.mkdir(parents=True, exist_ok=True)

        return cls(data=data, model=model, backtest=backtest, report=report, raw=cfg)
