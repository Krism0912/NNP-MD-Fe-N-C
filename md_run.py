import os.path

from ase.md.langevin import Langevin
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, write, Trajectory
import json
import numpy as np
import torch
import sys
import glob
import toml
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

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="MD simulations drive by graph neural networks", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--init_traj",
        type=str,
        help="Path to start configurations",
    )
    parser.add_argument(
        "--start_indice",
        type=int,
        help="Indice of the start configuration",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        help="Where to find the models",
    )
    parser.add_argument(
        "--time_step",
        type=float,
        default=0.5,
        help="Time step of MD simulation",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5000000,
        help="Maximum steps of MD",
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        default=100000,
        help="Minimum steps of MD, raise error if not reached",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=350.0,
        help="Maximum time steps of MD",
    )
    parser.add_argument(
        "--fix_under",
        type=float,
        default=5.9,
        help="Fix atoms under the specified value",
    )
    parser.add_argument(
        "--dump_step",
        type=int,
        default=100,
        help="Fix atoms under the specified value",
    )
    parser.add_argument(
        "--print_step",
        type=int,
        default=1,
        help="Fix atoms under the specified value",
    )
    parser.add_argument(
        "--num_uncertain",
        type=int,
        default=1000,
        help="Stop MD when too many structures with large uncertainty are collected",
    )
    parser.add_argument(
        "--plumed",
        action="store_true",
        help="Use plumed for MD",
    )
    parser.add_argument(
        "--plumed_restart",
        action="store_true",
        help="Restart metadynamics or not",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for this run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Set which device to use for running MD e.g. 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="arguments.toml",
        help="Path to config file. e.g. 'arguments.toml'"
    )

    return parser.parse_args(arg_list)
def new():
    args=argparse.ArgumentParser()
    args.add_argument("--init_traj",type=str)
    args.init_traj = r"/share/home/krism0912/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter6/Production/4site_5ns/FeNC_320H2O_4site.traj"
    args.start_indice= 0
    args.load_model=r"/share/home/krism0912/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter6/Ensemble"
    args.root=r"/share/home/krism0912/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter6/Production/4site_5ns"
    args.time_step= 1.0
    args.max_steps = 5 * 10**6 #5ns
    args.min_steps = 10**5
    args.temperature = 300
    args.fix_under =  False 
    args.wall_z = -1 
    args.dump_step = 50
    args.print_step = 1
    args.num_uncertain = 200000
    args.plumed = False
    args.plumed_restart = False
    args.random_seed = random.randint(10**8,10**9)
    args.device = 'cuda'
    args.cfg=None

    args.collect_interval= 0
    args.fix_beyond= 2.0
    args.warning_threshold= 0.03
    args.warning_threshold_max = 0.3
    args.stop_threshold= 0.1
    args.stop_threshold_max = 10


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
    # args = get_arguments()
    args=new()
    args.out_dir=os.path.join(args.root,'MD_RUN_{}'.format(RUN_iter))
    os.makedirs(args.out_dir, exist_ok=True)
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
        update_namespace(args, params)

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

    if args.plumed:
        from ase.calculators.plumed import Plumed
        setup = open(os.path.join(args.root,"plumed.dat"), "r").read().splitlines()
        # print(setup)
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
    else:
        atoms.calc = encalc



    collect_traj = Trajectory(os.path.join(args.out_dir,'warning_struct.traj'), 'w')
    out_e=open(os.path.join(args.out_dir, "energy_predict.txt"), 'w')
    pre_collect_step=50
    collect_num=0

    @CallsCounter
    def printenergy(a=atoms):  # store a reference to atoms in the definition.

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
        energy_predict=ensemble['energy_predict']
        out_e.write('Step {}  Energy: {}\n'.format(printenergy.calls * args.print_step,str(energy_predict)))
        nonlocal pre_collect_step, collect_num,exit_flag

        if forces_sd > args.stop_threshold or max_force_sd>args.stop_threshold_max:

            now_step= printenergy.calls * args.print_step
            pre_collect_step = now_step


            # if now_step - pre_collect_step >= args.collect_interval:
            #     pre_collect_step = now_step
            #     collect_num+=1
            #     if args.plumed:
            #         new_atoms = a.copy()
            #         new_atoms.calc = a.calc.calc
            #         collect_traj.write(new_atoms)
            #     else:
            #         collect_traj.write(a)


            logger.warning(
                "Steps= {:10d} Epot= {:12.3f} Ekin= {:12.3f} temperature= {:8.2f} energy_var= {:10.6f} forces_var= {:10.6f} forces_sd= {:10.6f} max_force_sd={:10.6f} forces_l2_var= {:10.6f} warning structure= {}, collected_ structure= {}".format(
                    printenergy.calls * args.print_step,
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
                 # sys.exit(0)
            else:
                exit_flag=True
                # sys.exit("Too large uncertainty!")
            return


        elif forces_sd > args.warning_threshold or max_force_sd > args.warning_threshold_max:
            now_step= printenergy.calls * args.print_step

            if now_step - pre_collect_step >= args.collect_interval:
                pre_collect_step = now_step
                collect_num+=1
                if args.plumed:
                    new_atoms = a.copy()
                    new_atoms.calc = a.calc.calc
                    collect_traj.write(new_atoms)
                else:
                    collect_traj.write(a)

            logger.warning("Steps= {:10d} Epot= {:12.3f} Ekin= {:12.3f} temperature= {:8.2f} energy_var= {:10.6f} forces_var= {:10.6f} forces_sd= {:10.6f} max_force_sd = {:10.6f} forces_l2_var= {:10.6f} warning structure= {}, collected_ structure= {}".format(
                printenergy.calls * args.print_step,
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
                    # sys.exit(0)
                else:
                    exit_flag=True
                    logger.info(f"More than {args.num_uncertain} uncertain structures are collected!")
                    # sys.exit(f"More than {args.num_uncertain} uncertain structures are collected!")
                return
        else:
            logger.info("Steps= {:10d} Epot= {:12.3f} Ekin= {:12.3f} temperature= {:8.2f} energy_var= {:10.6f} forces_var= {:10.6f} forces_sd= {:10.6f} max_force_sd= {:10.6f} forces_l2_var= {:10.6f} ".format(
                printenergy.calls * args.print_step,
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

    dyn = Langevin(atoms, args.time_step * units.fs, temperature_K=args.temperature, friction=0.02)
    dyn.attach(printenergy, interval=args.print_step)


    traj = Trajectory(os.path.join(args.out_dir,'MD.traj'), 'w', atoms)
    def write_traj(a=atoms):
        new_atoms = a.copy()
        new_atoms.calc = a.calc.calc
        traj.write(new_atoms)

    def adjust_velocity(a=atoms):
        coords = a.get_positions()
        for i,coord in enumerate(coords):
            if args.wall_z !=-1 and args.wall_z<coord[2]<args.wall_z+1.0:
                a.arrays['momenta'][i][2]*=-1

    def write_xyz(a=atoms):
        write(os.path.join(args.out_dir,'MD.xyz'), a,append=True)

    if os.path.exists(os.path.join(args.out_dir,'MD.xyz')):
        os.remove(os.path.join(args.out_dir,'MD.xyz'))
    dyn.attach(write_xyz if not args.plumed else write_traj, interval=args.dump_step)

    exit_flag=False
    for cnt in range(args.max_steps):
        if exit_flag:
            out_e.close()
            return collect_num,cnt
        try:
            dyn.run(1)
        except Exception as e:
            exit_flag=True
            pass
    # dyn.run(args.max_steps)
    return collect_num, args.max_steps


if __name__ == "__main__":
    args=new()
    collect=0
    RUN_iter=0
    os.makedirs(args.root, exist_ok=True)
    f = open(os.path.join(args.root, "MD_INFO.txt"), "w")
    f.write('Start NNP MD.\n')
    f.close()
    while collect<200:
        f = open(os.path.join(args.root, "MD_INFO.txt"), "a")
        RUN_iter+=1
        new_collect,steps=main(RUN_iter)
        collect+=new_collect
        f.write('RUN {}   total collect={}  steps={}\n'.format(RUN_iter,collect,steps))
        f.close()

