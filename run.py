import os
import pathlib
import shutil

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import tqdm

from src import siesta_io
from src import units
from src import getters


def _copy_examples(example_path):
    def _copy_all_files_in_work_dir(src):
        work_dir = pathlib.Path(os.getcwd())
        example_dir = pathlib.Path(hydra.utils.to_absolute_path(src))
        assert example_dir.exists()
        for f in example_dir.glob('*'):
            shutil.copy(f, work_dir)

    _copy_all_files_in_work_dir('examples/common')
    _copy_all_files_in_work_dir(f'examples/{example_path}')


def _max_difference(matrix1, matrix2):
    return np.amax(np.abs(matrix1 - matrix2))


@hydra.main(config_path='configs', config_name='run')
def app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Experiment settings
    _copy_examples(cfg.problem)
    optimizer_initializer = getters.get_optimizer(**cfg.optimizer)
    siesta_loc = cfg.exe 
    print(siesta_loc)
    cmd = f'{siesta_loc} < RUN.fdf  > stdout.txt'
    dH_history, dD_history = [], []

    # Loop
    total_loop = range(cfg.max_iters)
    if cfg.verbose:
        total_loop = tqdm.tqdm(total_loop)

    for iscf in total_loop:
        # Run SIESTA
        os.system(cmd)

        # Read calcuated Density Matrix, Hamiltonian Matrix
        nb1, ns1, numd1, listdptr1, listd1, D = siesta_io.readDM('D_new')
        nb2, ns2, numd2, listdptr2, listd2, H = siesta_io.readDM('H_new')

        # First step for iteration
        if iscf == 0:
            nb3, ns3, numd3, listdptr3, listd3, Dold = siesta_io.readDM('D_old')
            nb4, ns4, numd4, listdptr4, listd4, Hold = siesta_io.readDM('H_old')

            # Initialize torch tensor & optimizer
            s = torch.zeros_like(torch.Tensor(H))
            s.grad = torch.zeros_like(torch.Tensor(H))
            optimizer = optimizer_initializer(params=[s])
            f = open('log.txt', 'w')

        # Convergence test
        dH = _max_difference(Hold, H) / units.eV
        dD = _max_difference(Dold, D)
        dH_history.append(dH)
        dD_history.append(dD)

        if cfg.verbose:
            total_loop.set_description(f"dH: {dH:.8f} dD: {dD:.8f}")
            f.write(f"iscf: {iscf} dH: {dH:.8f} dD: {dD:.8f}\n")

        if (dH < cfg.hamiltonian_convergence_eps) \
            and (dD < cfg.density_convergence_eps):
            np.save('dH_history', np.array(dH_history))
            np.save('dD_history', np.array(dD_history))
            break

        # Mixing
        s[:] = torch.Tensor(H)
        s.grad[:] = torch.Tensor(H - Hold)
        optimizer.step()

        # Write mixed Hamiltonian matrix & density matrix for next iteration
        Hold[:] = s.numpy()
        Dold[:] = D
        siesta_io.writeDM('H_IN', nb2, ns2, numd2, listdptr2, listd2, s.numpy())

        # Save charge density and corresponding XC potential
        Rho = siesta_io.readGrid('RHO_in')
        Vxc = siesta_io.readGrid('Vxc_out')
        np.save(f'Rho_{iscf}', Rho)
        np.save(f'Vxc_{iscf}', Vxc)

    f.close()

if __name__ == "__main__":
    app()
