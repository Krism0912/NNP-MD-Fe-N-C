import os.path
from ase.md.langevin import Langevin
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, write, Trajectory
from ase.io.xyz import write_xyz
import json
import numpy as np
import torch
import sys
import glob
import argparse
from pathlib import Path
import logging

from PaiNN.data import AseDataset, collate_atomsdata
from PaiNN.model import PainnModel
from PaiNN.calculator import MLCalculator, EnsembleCalculator
from ase.constraints import FixAtoms
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

def new():
    args=argparse.ArgumentParser()
    args.add_argument("--init_traj",type=str)
    args.init_traj = r"/share/home/krism0912/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter8/Production/water_MTD_CV/last_frame.traj"
    args.start_indice= 0
    args.load_model=r"/share/home/krism0912/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter8/Ensemble"
    args.root= r"/share/home/krism0912/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter8/Production/water_MTD_CV"
    args.time_step= 1.0
    args.max_steps = 15000*1000 #15ns
    args.min_steps = 10**5
    args.temperature = 300
    args.fix_under =  False
    args.wall_z = -1
    args.dump_step = 100
    args.check_step = 1
    args.print_step = 20
    args.num_uncertain = 10000
    args.plumed = True
    args.plumed_restart = False
    args.random_seed = random.randint(10**8,10**9)
    args.device = 'cuda'
    args.cfg=None

    args.collect_interval= 100
    args.fix_beyond= None
    args.warning_threshold= 0.03
    args.warning_threshold_max = 0.4
    args.stop_threshold= 10
    args.stop_threshold_max = 100


    return args
def update_namespace(ns, d):
    for k, v in d.items():
        ns.__dict__[k] = v


class CallsCounter:
    def __init__(self, func):
        self.calls = 0
        self.func = func
    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.func(*args, **kwargs)

def main(RUN_iter):
    args=new()
    args.out_dir=os.path.join(args.root,'MD_RUN_{}'.format(RUN_iter))
    os.makedirs(args.out_dir, exist_ok=True)
    setup_seed(args.random_seed)

    # set logger

    logger = logging.getLogger(__file__)
    for handler in logger.handlers[:]:  # Iterate over a copy of the list of handlers
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.DEBUG)

    runHandler = logging.FileHandler(os.path.join(args.out_dir,'md.log'), mode='w')
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    errorHandler = logging.FileHandler(os.path.join(args.out_dir,'error.log'), mode='w')
    errorHandler.setLevel(logging.WARNING)
    errorHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))

    logger.addHandler(runHandler)
    logger.addHandler(errorHandler)
    logger.addHandler(logging.StreamHandler())
    logger.warning = CallsCounter(logger.warning)
    logger.info = CallsCounter(logger.info)

    # load model
    model_pth = Path(args.load_model).rglob('best_model*.pth')
    models = []
    for each in model_pth:
        # print(each)
        state_dict = torch.load(each)
        model = PainnModel(
            num_interactions=state_dict["num_layer"],
            hidden_state_size=state_dict["node_size"],
            cutoff=state_dict["cutoff"],
        )
        model.to(args.device)
        model.load_state_dict(state_dict["model"])

        models.append(model)

    encalc = EnsembleCalculator(models)

    # set up md start configuration
    images = Trajectory(args.init_traj)
    start_indice = np.random.choice(len(images)) if args.start_indice == None else args.start_indice
    logger.debug(f'Iter No.{RUN_iter}\nMD starts from No.{start_indice} configuration in {args.init_traj}')

    logger.info( 'warning_threshold= {} warning_threshold_max = {} \n '
                 'args.stop_threshold= {} args.stop_threshold_max = {}'.format(args.warning_threshold,
                                                                                  args.warning_threshold_max,
                                                                                  args.stop_threshold,
                                                                                  args.stop_threshold_max)
                 )
    atoms = images[start_indice]
    atoms.wrap()
    cons = FixAtoms(mask= ( atoms.positions[:, 2] > args.fix_beyond )& (atoms.positions[:, 2] < args.fix_under)) if args.fix_under  else []
    atoms.set_constraint(cons)

    from ase.calculators.plumed import Plumed
    setup = open(os.path.join(args.root,"plumed.dat"), "r").read().splitlines()

    setup[-2] = setup[-2].replace('HILL_PATH',f'{args.out_dir}' + '/HILL')
    setup[-1] = setup[-1].replace('COLVAR_PATH', f'{args.out_dir}' + '/COLVAR')


    atoms.calc = Plumed(
        calc=encalc,
        input=setup,
        timestep=args.time_step / units.fs,  # unit change, the first line of plumed must use TIME=fs
        atoms=atoms,
        kT=args.temperature * units.kB,
        restart=args.plumed_restart,
    )
    if args.plumed_restart:
        atoms.calc.istep = start_indice if start_indice >= 0 else len(images) + start_indice



    collect_traj = Trajectory(os.path.join(args.out_dir,'warning_struct.traj'), 'w')
    pre_collect_step=50
    collect_num=0

    @CallsCounter
    def checkenergy(a=atoms):  # store a reference to atoms in the definition.
        #Function to print the potential, kinetic and total energy.
        epot = a.get_potential_energy() if not args.plumed else a.get_potential_energy()[0]
        ekin = a.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB) / a.get_global_number_of_atoms()
        ensemble = a.calc.results['ensemble'] if not args.plumed else a.calc.calc.results['ensemble']
        energy_var = ensemble['energy_var']
        forces_var = np.mean(ensemble['forces_var'])
        forces_sd = np.mean(np.sqrt(ensemble['forces_var']))
        max_force_sd = np.max(np.sqrt(ensemble['forces_var']))
        forces_l2_var = np.mean(ensemble['forces_l2_var'])
        now_step = checkenergy.calls * args.check_step
        nonlocal pre_collect_step, collect_num,exit_flag
        if forces_sd > args.stop_threshold or max_force_sd>args.stop_threshold_max:
            pre_collect_step = now_step

            logger.warning(
                "Steps= {:10d} Epot= {:12.3f} Ekin= {:12.3f} temperature= {:7.1f} energy_var= {:10.3f} forces_var= {:10.3f} forces_sd= {:10.3f} max_force_sd = {:10.3f} forces_l2_var= {:10.3f} warning structure= {}, collected_ structure= {}".format(
                    checkenergy.calls * args.check_step,
                    epot,
                    ekin,
                    temp,
                    energy_var,
                    forces_var,
                    forces_sd,
                    max_force_sd,
                    forces_l2_var,
                    logger.warning.calls + 1,
                    collect_num
                ))
            logger.warning("Too large uncertainty!")
            if logger.info.calls + logger.warning.calls > args.min_steps:
                exit_flag = True
            else:
                exit_flag=True
            return


        elif forces_sd > args.warning_threshold or max_force_sd > args.warning_threshold_max:
            if now_step - pre_collect_step >= args.collect_interval:
                pre_collect_step = now_step
                collect_num+=1
                if args.plumed:
                    new_atoms = a.copy()
                    new_atoms.calc = a.calc.calc
                    collect_traj.write(new_atoms)
                else:
                    collect_traj.write(a)

            logger.warning("Steps= {:10d} Epot= {:12.3f} Ekin= {:12.3f} temperature= {:7.1f} energy_var= {:8.4f} forces_var= {:8.4f} forces_sd= {:8.4f} max_force_sd = {:8.4f} forces_l2_var= {:8.4f} warning structure= {}, collected_ structure= {}".format(
                checkenergy.calls * args.check_step,
                epot,
                ekin,
                temp,
                energy_var,
                forces_var,
                forces_sd,
                max_force_sd,
                forces_l2_var,
                logger.warning.calls+1,
                collect_num
            ))
            if collect_num > args.num_uncertain:
                logger.error(f"More than {args.num_uncertain} uncertain structures are collected!")

                if logger.info.calls + logger.warning.calls > args.min_steps:
                    logger.info("Have run more than min= {} steps!".format(args.min_steps))
                    exit_flag=True
                else:
                    exit_flag=True
                    logger.info(f"More than {args.num_uncertain} uncertain structures are collected!")
                return
        elif now_step % args.print_step == 0:
            logger.info("Steps= {:10d} Epot= {:12.3f} Ekin= {:12.3f} temperature= {:7.1f} energy_var= {:10.3f} forces_var= {:10.3f} forces_sd= {:10.3f} max_force_sd={:10.3f} forces_l2_var= {:10.3f}".format(
                checkenergy.calls * args.check_step,
                epot,
                ekin,
                temp,
                energy_var,
                forces_var,
                forces_sd,
                max_force_sd,
                forces_l2_var
            ))

    if not np.any(atoms.get_momenta()):
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)

    dyn = Langevin(atoms, args.time_step * units.fs, temperature_K=args.temperature, friction=0.002) #2ps-1
    dyn.attach(checkenergy, interval=args.check_step)

    md_step = 0
    def count_steps():
        nonlocal md_step
        md_step += 1
    dyn.attach(count_steps,interval=1)

    def output_xyz():
        file = os.path.join(args.out_dir, 'MD.xyz')
        pos = atoms.get_positions()
        sym = atoms.get_chemical_symbols()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        with open(file, "a") as f:
            f.write(f"{len(atoms)}\n")
            cell_flat = " ".join(f"{x:.6f}" for x in cell.reshape(-1))
            pbc_str = " ".join("T" if x else "F" for x in pbc)

            f.write(
                f'Lattice="{cell_flat}" Properties=species:S:1:pos:R:3 '
                f'pbc="{pbc_str}"\n'
            )
            for s, p in zip(sym, pos):
                f.write(f"{s} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

    dyn.attach(output_xyz, interval=args.dump_step)
    for file in ['MD.xyz']:
        if os.path.exists(os.path.join(args.out_dir,file)):
            os.remove(os.path.join(args.out_dir,file))
            
    exit_flag=False
    for cnt in range(args.max_steps):
        if exit_flag:
            return collect_num,cnt
        try:
            dyn.run(1)
        except Exception as e:
            exit_flag=True
            pass

    return collect_num, args.max_steps


if __name__ == "__main__":
    args=new()
    collect=0
    RUN_iter=0
    os.makedirs(args.root, exist_ok=True)
    f = open(os.path.join(args.root, "MD_INFO.txt"), "w")
    f.write('Start NNP MD.\n')
    f.close()
    while collect<500:
        f = open(os.path.join(args.root, "MD_INFO.txt"), "a")
        RUN_iter+=1
        new_collect,steps=main(RUN_iter)
        collect+=new_collect
        f.write('RUN {}   total collect={}  steps={}\n'.format(RUN_iter,collect,steps))
        f.close()

