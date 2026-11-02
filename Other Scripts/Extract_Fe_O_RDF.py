import sys
import numpy as np
from Calculate_single_RDF import  compute_rdf_A,compute_rdf_A_B

from basic import get_slab
def compute_rdf(traj,start,end,gap, zmin=-1, zmax=-1):
    n=len(traj)
    typeA='Fe'
    typeB='O'
    total_rdf=np.array([])
    cnt=0

    out.write('start: {} end: {} gap: {}\n'.format(start, min(n, end), gap))

    for idx in range(start,end,gap):
        print('Calculating ...:',idx)
        atoms=traj[idx]
        positions = atoms.get_positions(wrap=True)
        ele = np.array(atoms.get_chemical_symbols())
        box_size = atoms.cell.cellpar()[:3]
        positions_A = positions[ele == typeA] if zmin==-1 else positions[np.logical_and( ele == typeA, np.logical_and(positions[:, 2]>= zmin, positions[:, 2] <= zmax))]
        positions_B= positions[ele == typeB]

        rdf, r = compute_rdf_A_B(positions_A,positions_B,box_size, bin_size=0.025, max_distance = 4.5)
        total_rdf=total_rdf+rdf if np.any(total_rdf) else rdf
        cnt+=1
    total_rdf/=cnt
    return total_rdf,r


from ase.io import  write,Trajectory
import numpy as np
import os


root = r"/mnt/e/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter6/Production/FeNC_big_rerun_OK/"
# root = '/mnt/e/Project_MNC4/Dataset/NVT/Fe_NC_AIMD_25ps/'
out_dir = os.path.join(root, 'Statistics')
os.makedirs(out_dir, exist_ok=True)
out_fig = os.path.join(out_dir,'RDF_Fe_O.png')
out=open(os.path.join(out_dir,'RDF_Fe_O.txt'),'w')

traj=Trajectory(os.path.join(root,"MD_RUN_1/MD_for_analysis.traj"))
# traj = Trajectory("/mnt/e/Project_MNC4/Dataset/NVT/Fe_NC_AIMD_25ps/vasp_for_ase_out_all.traj")
n=len(traj)
zmin = -1
zmax = -1
rdf, r = compute_rdf(traj,0,n, 1,zmin, zmax)

print(*r)
print(*rdf)

from matplotlib import pyplot as plt
rdf[0]=0
plt.plot(r, rdf)

plt.xlabel('r')
plt.ylabel('g(r)')
# plt.legend()
plt.title('Radial Distribution Function (RDF)')
plt.savefig(out_fig)
plt.show()

out.write('r    g(r)\n')
for r,gr in zip(r,rdf):
    out.write('{}    {}\n'.format(r,gr))
out.close()


