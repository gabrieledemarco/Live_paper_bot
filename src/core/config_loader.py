"""Centralised configuration loader.

Reads `config.ini` and exposes a typed `PipelineConfig` dataclass that is
passed across all modules. Having a single place that interprets the INI
file keeps the rest of the codebase parameter-agnostic.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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


# --------------------------------------------------------------------------- #
# High-Timeframe (HTF) multi-timeframe track.
# These sections are OPTIONAL: when [HTF_DATA] is absent from the INI the
# tick-level OFI pipeline keeps working exactly as before and ``cfg.htf_*``
# attributes are ``None``.
# --------------------------------------------------------------------------- #
@dataclass
class HtfDataCfg:
    exchange: str
    pairs: List[str]
    timeframes: List[str]
    base_timeframe: str
    lookback_days: int
    output_dir: Path


@dataclass
class HtfFeatureCfg:
    vol_window: int
    vwap_window: int
    enrich_from_ticks: bool


@dataclass
class HtfModelCfg:
    task_type: str          # "classification" | "regression"
    model_type: str         # "LightGBM" | "Ridge"
    target_horizon: int     # base-timeframe bars ahead for the label
    threshold_bps: float    # flat-zone half-width (classification only)
    n_splits: int
    embargo: int            # bars purged between train/test folds
    top_k_features: int     # 0 = keep all
    holdout_frac: float     # fraction of the (time-ordered) tail used as test
    model_dir: Path


@dataclass
class HtfBacktestCfg:
    initial_capital: float
    leverage: float
    stop_loss_bps: float
    take_profit_bps: float
    taker_fee: float
    maker_fee: float
    maintenance_margin: float
    signal_threshold: float
    optimize_sltp: bool
    opt_sampler: str
    opt_n_trials: int
    opt_objective: str
    sl_bps_min: float
    sl_bps_max: float
    tp_bps_min: float
    tp_bps_max: float
    fee_grid: List[float]
    window: int
    meta_thr: float


@dataclass
class HtfStrategyV2Cfg:
    label_horizon: int
    ref_sl_bps: float
    ref_tp_bps: float
    entry_mode: str
    leverage: float
    thr_min: float
    thr_max: float
    opt_n_trials: int
    size_by_confidence: bool


@dataclass
class PipelineConfig:
    data: DataCfg
    model: ModelCfg
    backtest: BacktestCfg
    report: ReportCfg
    raw: configparser.ConfigParser = field(repr=False)
    htf_data: Optional[HtfDataCfg] = None
    htf_features: Optional[HtfFeatureCfg] = None
    htf_model: Optional[HtfModelCfg] = None
    htf_backtest: Optional[HtfBacktestCfg] = None
    htf_strategy_v2: Optional[HtfStrategyV2Cfg] = None

    @classmethod
    def load(cls, path: str | Path = "config.ini") -> "PipelineConfig":
        cfg = configparser.ConfigParser()
        config_path = Path(path).resolve()
        read = cfg.read(config_path)
        if not read:
            raise FileNotFoundError(f"Config file not found: {path}")

        # Project root = directory holding config.ini. ALL relative paths in
        # the INI are anchored here, NOT to the process working directory.
        # This makes ingestion (e.g. from the Streamlit "download" button) and
        # later reads (training/backtest) always hit the same folders, even
        # when the CWD differs between Streamlit reruns / launch contexts.
        root = config_path.parent

        def _resolve(p: str) -> Path:
            q = Path(p)
            return q if q.is_absolute() else (root / q)

        data = DataCfg(
            input_dir=_resolve(cfg.get("DATA", "input_dir")),
            output_dir=_resolve(cfg.get("DATA", "output_dir")),
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
            model_dir=_resolve(cfg.get("MODEL", "model_dir")),
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
            charts_dir=_resolve(cfg.get("REPORT", "charts_dir")),
            log_dir=_resolve(cfg.get("REPORT", "log_dir")),
        )

        # Ensure mandatory output dirs exist.
        for p in (data.output_dir, model.model_dir, report.charts_dir, report.log_dir):
            p.mkdir(parents=True, exist_ok=True)

        # ---- Optional HTF multi-timeframe track --------------------------- #
        htf_data = htf_features = htf_model = None
        if cfg.has_section("HTF_DATA"):
            htf_data = HtfDataCfg(
                exchange=cfg.get("HTF_DATA", "exchange", fallback="binance"),
                pairs=[p.strip().upper() for p in cfg.get("HTF_DATA", "pairs").split(",")],
                timeframes=[t.strip() for t in cfg.get("HTF_DATA", "timeframes").split(",")],
                base_timeframe=cfg.get("HTF_DATA", "base_timeframe", fallback="1m"),
                lookback_days=cfg.getint("HTF_DATA", "lookback_days", fallback=180),
                output_dir=_resolve(cfg.get("HTF_DATA", "output_dir", fallback="data/ohlcv")),
            )
            htf_features = HtfFeatureCfg(
                vol_window=cfg.getint("HTF_FEATURES", "vol_window", fallback=20),
                vwap_window=cfg.getint("HTF_FEATURES", "vwap_window", fallback=20),
                enrich_from_ticks=cfg.getboolean("HTF_FEATURES", "enrich_from_ticks", fallback=True),
            )
            htf_model = HtfModelCfg(
                task_type=cfg.get("HTF_MODEL", "task_type", fallback="classification").lower(),
                model_type=cfg.get("HTF_MODEL", "model_type", fallback="LightGBM"),
                target_horizon=cfg.getint("HTF_MODEL", "target_horizon", fallback=5),
                threshold_bps=cfg.getfloat("HTF_MODEL", "threshold_bps", fallback=5.0),
                n_splits=cfg.getint("HTF_MODEL", "n_splits", fallback=5),
                embargo=cfg.getint("HTF_MODEL", "embargo", fallback=10),
                top_k_features=cfg.getint("HTF_MODEL", "top_k_features", fallback=0),
                holdout_frac=cfg.getfloat("HTF_MODEL", "holdout_frac", fallback=0.2),
                model_dir=model.model_dir,
            )
            htf_data.output_dir.mkdir(parents=True, exist_ok=True)

        # ---- Optional HTF cost-aware backtest ----------------------------- #
        htf_backtest = None
        if cfg.has_section("HTF_BACKTEST"):
            htf_backtest = HtfBacktestCfg(
                initial_capital=cfg.getfloat("HTF_BACKTEST", "initial_capital", fallback=10000.0),
                leverage=cfg.getfloat("HTF_BACKTEST", "leverage", fallback=3.0),
                stop_loss_bps=cfg.getfloat("HTF_BACKTEST", "stop_loss_bps", fallback=10.0),
                take_profit_bps=cfg.getfloat("HTF_BACKTEST", "take_profit_bps", fallback=20.0),
                taker_fee=cfg.getfloat("HTF_BACKTEST", "taker_fee", fallback=0.0004),
                maker_fee=cfg.getfloat("HTF_BACKTEST", "maker_fee", fallback=0.0002),
                maintenance_margin=cfg.getfloat("HTF_BACKTEST", "maintenance_margin", fallback=0.005),
                signal_threshold=cfg.getfloat("HTF_BACKTEST", "signal_threshold", fallback=0.45),
                optimize_sltp=cfg.getboolean("HTF_BACKTEST", "optimize_sltp", fallback=True),
                opt_sampler=cfg.get("HTF_BACKTEST", "opt_sampler", fallback="gp").lower(),
                opt_n_trials=cfg.getint("HTF_BACKTEST", "opt_n_trials", fallback=50),
                opt_objective=cfg.get("HTF_BACKTEST", "opt_objective", fallback="sharpe").lower(),
                sl_bps_min=cfg.getfloat("HTF_BACKTEST", "sl_bps_min", fallback=3.0),
                sl_bps_max=cfg.getfloat("HTF_BACKTEST", "sl_bps_max", fallback=40.0),
                tp_bps_min=cfg.getfloat("HTF_BACKTEST", "tp_bps_min", fallback=3.0),
                tp_bps_max=cfg.getfloat("HTF_BACKTEST", "tp_bps_max", fallback=80.0),
                fee_grid=[float(x) for x in cfg.get("HTF_BACKTEST", "fee_grid", fallback="0.5,1,2").split(",")],
                window=cfg.getint("HTF_BACKTEST", "window", fallback=8),
                meta_thr=cfg.getfloat("HTF_BACKTEST", "meta_thr", fallback=0.55),
            )

        # ---- Optional HTF strategy v2 ------------------------------------- #
        htf_strategy_v2 = None
        if cfg.has_section("HTF_STRATEGY_V2"):
            htf_strategy_v2 = HtfStrategyV2Cfg(
                label_horizon=cfg.getint("HTF_STRATEGY_V2", "label_horizon", fallback=30),
                ref_sl_bps=cfg.getfloat("HTF_STRATEGY_V2", "ref_sl_bps", fallback=15.0),
                ref_tp_bps=cfg.getfloat("HTF_STRATEGY_V2", "ref_tp_bps", fallback=30.0),
                entry_mode=cfg.get("HTF_STRATEGY_V2", "entry_mode", fallback="maker").lower(),
                leverage=cfg.getfloat("HTF_STRATEGY_V2", "leverage", fallback=2.0),
                thr_min=cfg.getfloat("HTF_STRATEGY_V2", "thr_min", fallback=0.50),
                thr_max=cfg.getfloat("HTF_STRATEGY_V2", "thr_max", fallback=0.75),
                opt_n_trials=cfg.getint("HTF_STRATEGY_V2", "opt_n_trials", fallback=60),
                size_by_confidence=cfg.getboolean("HTF_STRATEGY_V2", "size_by_confidence", fallback=True),
            )

        return cls(data=data, model=model, backtest=backtest, report=report, raw=cfg,
                   htf_data=htf_data, htf_features=htf_features, htf_model=htf_model,
                   htf_backtest=htf_backtest, htf_strategy_v2=htf_strategy_v2)
