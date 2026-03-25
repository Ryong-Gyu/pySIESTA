# pySIESTA

`pySIESTA` is a Python-based driver that controls a local SIESTA self-consistent mixing loop by reading SIESTA outputs (`H_new`, `D_new`) and iteratively mixing Hamiltonians.

## Features

- YAML-based runtime configuration (`configs/default.yaml`)
- CLI argument overrides for runtime settings
- Selectable linear or Pulay mixer
- Convergence checks on both `dH` and `dD`
- Convergence history export (`dH_history.npy`, `dD_history.npy`)

## Repository Layout

```text
pySIESTA/
├── pysiesta/
│   ├── app.py                # Main execution loop
│   ├── config.py             # Config loading + CLI arguments
│   ├── __main__.py           # Entry point for `python -m pysiesta`
│   └── utils/
│       ├── siesta_io.py      # SIESTA binary I/O helpers
│       └── optimizer/        # linear and pulay mixers
├── configs/
│   ├── default.yaml
│   └── optimizer/
├── examples/
└── run.py                    # Script entry point
```

## Installation

Use Python 3.9+.

```bash
pip install -e .
```

## Usage

### Basic run

Your working directory should contain `RUN.fdf` and required SIESTA files (`H_old`, `D_old`, `H_new`, `D_new`).

```bash
python -m pysiesta --config configs/default.yaml
```

Or:

```bash
python run.py --config configs/default.yaml
```

### Common CLI overrides

```bash
python -m pysiesta \
  --max-iters 80 \
  --hamiltonian-convergence-eps 1e-5 \
  --density-convergence-eps 1e-4 \
  --optimizer-name pulay \
  --optimizer-alpha 0.5 \
  --optimizer-history-size 6 \
  --optimizer-regularization 1e-10 \
  --nproc 8 \
  --exe /path/to/siesta
```

## Example configuration

```yaml
max_iters: 50
hamiltonian_convergence_eps: 1e-4
density_convergence_eps: 1e-3
exe: ""
nproc: 8
optimizer:
  name: linear
  alpha: 0.5
  history_size: 6
  regularization: 1e-10
```

## Outputs

- `log.txt`: per-iteration `dH` and `dD` values
- `dH_history.npy`, `dD_history.npy`: convergence history arrays
- `stdout.txt`: SIESTA standard output

## License

Unless otherwise noted, this project is intended for internal research/development usage.
