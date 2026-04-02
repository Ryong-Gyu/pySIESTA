from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np

from pysiesta.config import add_cli_arguments, load_config, refresh_config
from pysiesta.utils import getters
from pysiesta.utils import siesta_io
from pysiesta.utils import units


script_path = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_EXE = f"{script_path}/pysiesta/utils/siesta"

def _max_difference(matrix1, matrix2):
    return np.amax(np.abs(matrix1 - matrix2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local SIESTA self-consistent mixing loop")
    add_cli_arguments(parser)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    base_config = load_config(args.config)
    config = refresh_config(base=base_config, args=args)

    exe = config.exe or DEFAULT_EXE

    work_dir = pathlib.Path.cwd()
    run_fdf = work_dir / "RUN.fdf"
    if not run_fdf.exists():
        raise FileNotFoundError(f"RUN.fdf not found at {run_fdf}.")

    mixer_initializer = getters.get_mixer(
        name=config.optimizer.name,
        alpha=config.optimizer.alpha,
        history_size=config.optimizer.history_size,
        regularization=config.optimizer.regularization,
    )
    mixer = mixer_initializer()

    cmd = f"mpirun -np {config.nproc} {exe} < RUN.fdf >> stdout.txt 2>&1"
    dH_history, dD_history = [], []

    # initial running
    os.system(cmd)
    nb3, ns3, numd3, listdptr3, listd3, Dold = siesta_io.readDM("D_old")
    nb4, ns4, numd4, listdptr4, listd4, Hold = siesta_io.readDM("H_old")

    with open("log.txt", "w", encoding="utf-8") as f:
        for iscf in range(1, config.max_iters + 1):
            os.system(cmd)

            nb1, ns1, numd1, listdptr1, listd1, D = siesta_io.readDM("D_new")
            nb2, ns2, numd2, listdptr2, listd2, H = siesta_io.readDM("H_new")

            dH = _max_difference(Hold, H) / units.eV
            dD = _max_difference(Dold, D)
            dH_history.append(dH)
            dD_history.append(dD)

            print(f"iscf: {iscf} dH: {dH:.8f} dD: {dD:.8f}")
            f.write(f"iscf: {iscf} dH: {dH:.8f} dD: {dD:.8f}\n")
            os.system(f"echo '{iscf} dH: {dH:.8f} dD: {dD:.8f}' >> file.txt")

            if (dH < config.hamiltonian_convergence_eps) and (dD < config.density_convergence_eps):
                np.save("dH_history", np.array(dH_history))
                np.save("dD_history", np.array(dD_history))
                break

            mixed_hamiltonian = mixer.mix(x_out=H, x_in=Hold)

            Hold[:] = mixed_hamiltonian
            Dold[:] = D
            siesta_io.writeDM("H_IN", nb2, ns2, numd2, listdptr2, listd2, mixed_hamiltonian)


if __name__ == "__main__":
    main()
