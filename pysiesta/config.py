from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required to read configuration files.") from exc


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "linear"
    alpha: float = 0.5
    history_size: int = 6
    regularization: float = 1e-10


@dataclass(frozen=True)
class RunConfig:
    max_iters: int = 50
    hamiltonian_convergence_eps: float = 1e-4
    density_convergence_eps: float = 1e-3
    exe: str = ""
    nproc: int = 8
    optimizer: OptimizerConfig = OptimizerConfig()


DEFAULT_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return data


def load_config(config_path: str | pathlib.Path | None = None) -> RunConfig:
    path = pathlib.Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw = _load_yaml(path)

    optimizer_raw = raw.get("optimizer", {}) or {}

    optimizer = OptimizerConfig(
        name=str(optimizer_raw.get("name", "linear")),
        alpha=float(optimizer_raw.get("alpha", 0.5)),
        history_size=int(optimizer_raw.get("history_size", 6)),
        regularization=float(optimizer_raw.get("regularization", 1e-10)),
    )

    return RunConfig(
        max_iters=int(raw.get("max_iters", 50)),
        hamiltonian_convergence_eps=float(raw.get("hamiltonian_convergence_eps", 1e-4)),
        density_convergence_eps=float(raw.get("density_convergence_eps", 1e-3)),
        exe=str(raw.get("exe", "")),
        nproc=int(raw.get("nproc", 8)),
        optimizer=optimizer,
    )


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config")

    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--hamiltonian-convergence-eps", type=float)
    parser.add_argument("--density-convergence-eps", type=float)

    parser.add_argument("--optimizer-name", type=str)
    parser.add_argument("--optimizer-alpha", type=float)
    parser.add_argument("--optimizer-history-size", type=int)
    parser.add_argument("--optimizer-regularization", type=float)

    parser.add_argument("--exe", type=str)
    parser.add_argument("--nproc", type=int)


def refresh_config(base: RunConfig, args: argparse.Namespace) -> RunConfig:
    optimizer = OptimizerConfig(
        name=args.optimizer_name if args.optimizer_name is not None else base.optimizer.name,
        alpha=args.optimizer_alpha if args.optimizer_alpha is not None else base.optimizer.alpha,
        history_size=args.optimizer_history_size if args.optimizer_history_size is not None else base.optimizer.history_size,
        regularization=args.optimizer_regularization
        if args.optimizer_regularization is not None
        else base.optimizer.regularization,
    )

    return RunConfig(
        max_iters=args.max_iters if args.max_iters is not None else base.max_iters,
        hamiltonian_convergence_eps=args.hamiltonian_convergence_eps
        if args.hamiltonian_convergence_eps is not None
        else base.hamiltonian_convergence_eps,
        density_convergence_eps=args.density_convergence_eps
        if args.density_convergence_eps is not None
        else base.density_convergence_eps,
        exe=args.exe if args.exe is not None else base.exe,
        nproc=args.nproc if args.nproc is not None else base.nproc,
        optimizer=optimizer,
    )
