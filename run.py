import argparse
import os
import pathlib
import shutil

import numpy as np
import torch
import tqdm

from src import getters
from src import siesta_io
from src import units


DEFAULT_EXE = "/home2/rong/00.Development/0.GitHub/8.KS-DFT_local/siesta"


def _copy_examples(example_path: str, work_dir: pathlib.Path, base_dir: pathlib.Path) -> None:
    def _copy_all_files_in_work_dir(src: str) -> None:
        example_dir = base_dir / src
        if not example_dir.exists():
            raise FileNotFoundError(f"Example directory not found: {example_dir}")
        for f in example_dir.glob("*"):
            shutil.copy(f, work_dir)

    _copy_all_files_in_work_dir("examples/common")
    _copy_all_files_in_work_dir(f"examples/{example_path}")


def _max_difference(matrix1, matrix2):
    return np.amax(np.abs(matrix1 - matrix2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local SIESTA self-consistent mixing loop")
    parser.add_argument("--work-dir", default=None, help="Working directory that contains RUN.fdf")
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--hamiltonian-convergence-eps", type=float, default=1e-5)
    parser.add_argument("--density-convergence-eps", type=float, default=1e-5)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--exe", default=DEFAULT_EXE, help="Path to SIESTA executable")

    parser.add_argument("--optimizer-name", default="SGD")
    parser.add_argument("--optimizer-lr", type=float, default=0.5)
    parser.add_argument("--optimizer-momentum", type=float, default=0.0)
    parser.add_argument("--optimizer-nesterov", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--optimizer-weight-decay", type=float, default=0.0)

    parser.add_argument(
        "--problem",
        default=None,
        help="Optional example path under examples/, e.g. molecular/0001. If set, example files are copied into work-dir.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    script_dir = pathlib.Path(__file__).resolve().parent
    work_dir = pathlib.Path(args.work_dir).resolve() if args.work_dir else script_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.problem:
        _copy_examples(args.problem, work_dir=work_dir, base_dir=script_dir)

    run_fdf = work_dir / "RUN.fdf"
    if not run_fdf.exists():
        raise FileNotFoundError(
            f"RUN.fdf not found at {run_fdf}. "
            "Put RUN.fdf in the working directory or use --problem to copy an example first."
        )

    os.chdir(work_dir)

    optimizer_initializer = getters.get_optimizer(
        name=args.optimizer_name,
        lr=args.optimizer_lr,
        momentum=args.optimizer_momentum,
        nesterov=args.optimizer_nesterov,
        weight_decay=args.optimizer_weight_decay,
    )

    cmd = f"{args.exe} < RUN.fdf  > stdout.txt"
    dH_history, dD_history = [], []

    total_loop = range(args.max_iters)
    if args.verbose:
        total_loop = tqdm.tqdm(total_loop)

    for iscf in total_loop:
        os.system(cmd)

        nb1, ns1, numd1, listdptr1, listd1, D = siesta_io.readDM("D_new")
        nb2, ns2, numd2, listdptr2, listd2, H = siesta_io.readDM("H_new")

        if iscf == 0:
            nb3, ns3, numd3, listdptr3, listd3, Dold = siesta_io.readDM("D_old")
            nb4, ns4, numd4, listdptr4, listd4, Hold = siesta_io.readDM("H_old")

            s = torch.zeros_like(torch.Tensor(H))
            s.grad = torch.zeros_like(torch.Tensor(H))
            optimizer = optimizer_initializer(params=[s])
            f = open("log.txt", "w")

        dH = _max_difference(Hold, H) / units.eV
        dD = _max_difference(Dold, D)
        dH_history.append(dH)
        dD_history.append(dD)

        if args.verbose:
            total_loop.set_description(f"dH: {dH:.8f} dD: {dD:.8f}")
            f.write(f"iscf: {iscf} dH: {dH:.8f} dD: {dD:.8f}\n")

        if (dH < args.hamiltonian_convergence_eps) and (dD < args.density_convergence_eps):
            np.save("dH_history", np.array(dH_history))
            np.save("dD_history", np.array(dD_history))
            break

        s[:] = torch.Tensor(H)
        s.grad[:] = torch.Tensor(H - Hold)
        optimizer.step()

        Hold[:] = s.numpy()
        Dold[:] = D
        siesta_io.writeDM("H_IN", nb2, ns2, numd2, listdptr2, listd2, s.numpy())

        Rho = siesta_io.readGrid("RHO_in")
        Vxc = siesta_io.readGrid("Vxc_out")
        np.save(f"Rho_{iscf}", Rho)
        np.save(f"Vxc_{iscf}", Vxc)

    f.close()


if __name__ == "__main__":
    main()
