"""
Command-line entry point.

    python run_experiments.py --config configs/ensemble_dpr.yaml
    python run_experiments.py --all
    python run_experiments.py --config configs/blr_dpr.yaml --set bnn.epochs=30

`--set` overrides any config key by its dotted path; the effective config is
saved in each run's summary.json.
"""

from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path

from .config import ExperimentConfig, load_config
from .pipeline import run_all, run_experiment
from .utils import configure_logging

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_CONFIGS = ("ensemble_pou.yaml", "ensemble_dpr.yaml", "blr_pou.yaml", "blr_dpr.yaml")


def _parse_override(item: str) -> dict:
    """Turn 'a.b.c=value' into {'a': {'b': {'c': value}}}, literal-parsing the value."""
    if "=" not in item:
        raise argparse.ArgumentTypeError(f"--set expects key=value, got '{item}'")
    key, raw = item.split("=", 1)
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw
    out: dict = value
    for part in reversed(key.split(".")):
        out = {part: out}
    return out


def _merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        a[k] = _merge(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_experiments",
        description="Decision-aware evaluation of uncertainty-based deferral.",
    )
    p.add_argument("--config", type=Path, action="append", help="config YAML (repeatable)")
    p.add_argument("--all", action="store_true", help=f"run the default suite: {', '.join(DEFAULT_CONFIGS)}")
    p.add_argument("--output-dir", type=str, default=None, help="override output_dir")
    p.add_argument("--project-root", type=str, default=None, help="root that data paths resolve against")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(getattr(logging, args.log_level), args.log_file)

    paths = list(args.config or [])
    if args.all or not paths:
        paths = [CONFIG_DIR / name for name in DEFAULT_CONFIGS]

    overrides: dict = {}
    for item in args.overrides:
        overrides = _merge(overrides, _parse_override(item))
    for key, value in (
        ("output_dir", args.output_dir),
        ("project_root", args.project_root),
        ("seed", args.seed),
    ):
        if value is not None:
            overrides[key] = value
    if args.no_plots:
        overrides = _merge(overrides, {"plots": {"enabled": False}})

    configs: list[ExperimentConfig] = [load_config(p, overrides) for p in paths]
    logger.info("Running %d experiment(s): %s", len(configs), ", ".join(c.name for c in configs))

    if len(configs) == 1:
        run_experiment(configs[0])
    else:
        run_all(configs)
    logger.info("Done. Outputs under %s", configs[0].output_dir)
    return 0
