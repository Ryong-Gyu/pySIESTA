import argparse
import os
import pathlib
import shutil
from typing import Optional

import numpy as np
import tqdm

from src import getters
from src import siesta_io
from src import units



script_path = pathlib.Path(__file__).resolve().parent
DEFAULT_EXE = f"{script_path}/src/siesta"


def _max_difference(matrix1, matrix2):
    return np.amax(np.abs(matrix1 - matrix2))

def _add_bool_argument(parser: argparse.ArgumentParser, name: str, default: bool, help: Optional[str] = None) -> None:

    def _str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    option = f"--{name}"
    parser.add_argument(option, type=_str2bool, nargs='?', const=True, default=default, help=help)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local SIESTA self-consistent mixing loop")

    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--hamiltonian-convergence-eps", type=float, default=1e-4)
    parser.add_argument("--density-convergence-eps", type=float, default=1e-3)

    parser.add_argument("--optimizer-name", default="linear", help="Mixer type: linear or pulay")
    parser.add_argument("--optimizer-alpha", type=float, default=0.5, help="Linear mixing factor")
    parser.add_argument("--optimizer-history-size", type=int, default=6, help="History size for Pulay mixer")
    parser.add_argument("--optimizer-regularization", type=float, default=1e-10, help="Ridge term for Pulay linear solve")

    parser.add_argument("--exe", default=DEFAULT_EXE, help="Path to SIESTA executable")
    parser.add_argument("--nproc", type=int, default=8)

    return parser

def main() -> None:
    args = _build_parser().parse_args()

    # Read RUN.fdf
    work_dir = pathlib.Path.cwd()
    run_fdf = work_dir / "RUN.fdf"
    if not run_fdf.exists():
        raise FileNotFoundError(
            f"RUN.fdf not found at {run_fdf}. "
            "Put RUN.fdf in the working directory or use --problem to copy an example first."
        )

    # Initialize SCF mixer
    mixer_initializer = getters.get_mixer(
        name=args.optimizer_name,
        alpha=args.optimizer_alpha,
        history_size=args.optimizer_history_size,
        regularization=args.optimizer_regularization,
    )

    cmd = f"mpirun -np {args.nproc} {args.exe} < RUN.fdf  > stdout.txt"
    dH_history, dD_history = [], []

    # SCF loop
    for iscf in range(1,args.max_iters+1):
        os.system(cmd)

        nb1, ns1, numd1, listdptr1, listd1, D = siesta_io.readDM("D_new")
        nb2, ns2, numd2, listdptr2, listd2, H = siesta_io.readDM("H_new")

        if iscf == 0:
            nb3, ns3, numd3, listdptr3, listd3, Dold = siesta_io.readDM("D_old")
            nb4, ns4, numd4, listdptr4, listd4, Hold = siesta_io.readDM("H_old")

            mixer = mixer_initializer()
            f = open("log.txt", "w")

        dH = _max_difference(Hold, H) / units.eV
        dD = _max_difference(Dold, D)
        dH_history.append(dH)
        dD_history.append(dD)

        print(f"iscf: {iscf} dH: {dH:.8f} dD: {dD:.8f}\n")
        f.write(f"iscf: {iscf} dH: {dH:.8f} dD: {dD:.8f}\n")

        if (dH < args.hamiltonian_convergence_eps) and (dD < args.density_convergence_eps):
            np.save("dH_history", np.array(dH_history))
            np.save("dD_history", np.array(dD_history))
            break

        mixed_hamiltonian = mixer.mix(x_out=H, x_in=Hold)

        Hold[:] = mixed_hamiltonian
        Dold[:] = D
        siesta_io.writeDM("H_IN", nb2, ns2, numd2, listdptr2, listd2, mixed_hamiltonian)

    f.close()


if __name__ == "__main__":
    main()
