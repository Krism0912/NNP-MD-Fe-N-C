from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.io.trajectory import Trajectory
import random
# Create your atomic structure
import logging
import os.path

dataset = r"/public4/home/sc54073/YD_run/Fe_NC_multi_site/Iter1/DFT_300K/warning_struct.traj"
dataset = Trajectory(dataset)
n=len(dataset)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler(
             "DFT_log.txt", mode="w"
        ),
        logging.StreamHandler(),
    ],
)

for idx in range(n):
    out_dir='./idx = {}'.format(idx)
    atoms=dataset[idx]
    logging.info('Calculating No.{}... Atoms.calc= {}'.format(idx,atoms.calc))
    # Define your custom INCAR file path


    # Set up the Vasp calculator
    atom=dataset[idx]
    atom.set_initial_magnetic_moments([0]*18+ [0]*4 +[2]*1+[0]*20 +[0]*40)

    calc = Vasp(atoms = atom,
                directory = out_dir,
                kpts = (3,2,2),  # Example k-points
	magmom = atom.get_initial_magnetic_moments(),
                )
    calc.set(ismear=0)
    calc.set(sigma=0.1)
    calc.set(lreal='AUTO')
    calc.set(ncore=7)
    calc.set(npar=8)
    # calc.set(maxmix=50)
    #calc.set(algo='Fast')
    calc.set(prec='Normal')
    calc.set(isym=0)
    calc.set(xc='rpbe')
    calc.set(encut=400)
    calc.set(ivdw=11)
    calc.set(nelm=400)
    calc.set(lwave=False)
    calc.set(lcharg=False)
    calc.set(ibrion=-1)
    calc.set(isif=2)
    calc.set(ispin=2)
    #calc.set(ediff=1e-5)
    calc.set(nupdown =2)
    
    

    #calc.write_input(dataset[0])
    # Attach the calculator to the atoms object
    atoms.calc = calc

    # Run your calculation
    energy = atoms.get_potential_energy()
    logging.info(f'Calculated energy: {energy} eV')