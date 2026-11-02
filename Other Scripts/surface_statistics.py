import sys

from ase.io import read,write,Trajectory
from ase.geometry import get_angles
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from basic import *
from collections import defaultdict
import sys



# Load the entire trajectory
root = r"/mnt/e/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter8/Production/4site_5ns_bilayer/"
in_=os.path.join(root,"MD_RUN_1/MD_for_analysis.traj")

frames=Trajectory(in_)

traj_size = len(frames)
box = frames[0].cell.cellpar()[:3] # NVT simulation box



mask = np.array([atom.index for atom in frames[0] if atom.symbol in ['C','N','Fe']])




dt = 0.05 # ps
def fun(start,end=len(frames),gap = 1):
    pos = [None] * end
    data = []
    for i in range(start,end,gap):
        print(i)
        # frames[i].positions[:,2] +=20
        frames[i].wrap()
        pos[i] = frames[i].get_positions()
        zs = pos[i][:,2][mask]
        zs = zs[zs >= 2.5]
        print(len(zs))
        variance = np.var(zs, ddof=0)
        data.append([i,variance])
    data=np.array(data)
    x = data[:, 0]*dt
    y = data[:, 1]
    plt.plot(x,y)
    plt.title('Var of surface')  #####
    plt.xlabel('time(ps)')
    plt.ylabel('variance')
    plt.tight_layout()
    plt.show()

    tmp_root = os.path.join(root,'Statistics/surface_statistics')
    os.makedirs(tmp_root,exist_ok=True)
    out = open(os.path.join(tmp_root, 'data.txt'), mode='w') #  更改!!!!!!!
    out.write('Start frame: {} End frame: {}  gap: {}\n'.format(start,end, gap))

    out.write('\n\n')
    out.write('time(ps) var  \n')
    for time,var in zip(x,y):
        out.write('{}   {}\n'.format(time,var))
    out.close()
    return data
fun(0,len(frames),10)