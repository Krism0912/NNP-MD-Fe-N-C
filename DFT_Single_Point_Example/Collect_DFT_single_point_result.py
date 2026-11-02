import os
from ase.io import read, write, Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np

root=r"/share/home/krism0912/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter4/DFT_300K"
out=os.path.join(root,'DFT_SP_calc.xyz')
if os.path.exists(out):
    os.remove(out)

out_traj=os.path.join(root,'DFT_SP_calc.traj')
filenames=os.listdir(root)
filenames=[x for x in filenames if x.startswith('idx')]
filenames.sort(key=lambda x:(len(x),x))
print(filenames)


from From_vasp_to_xyz import VASP_2_XYZ
convert=VASP_2_XYZ(root=None,out_file=out,write_interval_1=1,write_interval_2=1,format='DFT')
for folder in filenames:
    convert.root=os.path.join(root,folder)
    convert.E=[]
    convert.get_E()
    convert.read_atoms()
    convert.write(mode='a')

images = read(out, index=':')
n=len(images)
ok=[]
for i, atoms in enumerate(images):
    if i%500==0:
        print('Wrting ASE traj file... {:.2f}%,{}/{}'.format(100 * i / n, i,n))
    calc = SinglePointCalculator(atoms, energy=atoms.info.pop('TotEnergy'), forces=atoms.arrays.pop('force'))
    atoms.calc = calc
    print(i,np.max(atoms.calc.get_forces()))
    if np.max(atoms.calc.get_forces())>10.0:
        continue
    ok.append(atoms)

write(out_traj, ok)
print('Final size ',len(ok))
print('.traj file Generation complete!')



