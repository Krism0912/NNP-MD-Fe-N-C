import sys

from ase.io import read,write,Trajectory
from ase.geometry import get_angles
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from basic import pbc_distance,pbc_angle,get_slab,check_in_region,identify_water_molecules
from collections import defaultdict
import sys
from scipy.ndimage import gaussian_filter1d

# Load the entire trajectory
root = r"/mnt/e/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter6/Production/4site_5ns/"

data =  np.load(os.path.join(root, 'Statistics/res_time/idxs.npy'))

Fe_cnt = len(data)

all_tau = []

# =====================================
# Extract durations
# =====================================

gap_tol = 20
for site in range(Fe_cnt):

    arr = data[site]
    current = arr[0]

    count = 1

    gap_count = 0

    for i in range(1, len(arr)):

        # same water
        if arr[i] == current:

            count += 1
            gap_count = 0

        else:

            # check whether it returns
            returned = False

            for k in range(1, gap_tol + 1):

                if i + k < len(arr):

                    if arr[i + k] == current:

                        returned = True
                        break

            if returned:
                count += 1
                gap_count += 1

            else:
                all_tau.append(count)

                current = arr[i]

                count = 1
                gap_count = 0
    all_tau.append(count)
all_tau = [x for x in all_tau if x >= gap_tol] # 只统计 >1ps 的吸附行为，否则可能在 不稳定 exchange
print(all_tau)
# convert to ps
dt = 0.05

all_tau = np.array(all_tau) * dt

# =====================================
# Weighted survival probability
# =====================================

t_grid = np.linspace(
    0,
    800,
    200
)

survival_w = []

total_weight = np.sum(all_tau)

for t in t_grid:

    mask = all_tau > t

    s = np.sum(all_tau[mask]) / total_weight

    survival_w.append(s)

survival_w = np.array(survival_w)


from scipy.interpolate import PchipInterpolator

interp = PchipInterpolator(
    t_grid,
    survival_w
)

x_smooth = np.linspace(
    t_grid.min(),
    t_grid.max(),
    1000
)

y_smooth = interp(x_smooth)

plt.figure(dpi=600)

plt.plot(
    x_smooth,
    y_smooth,
    lw=2.5
)

plt.xlabel("Residence time (ps)")
plt.ylabel("Weighted survival probability")

plt.title("Occupancy-weighted residence survival")

plt.xlim(left=0)
plt.ylim(0, 1)

plt.tight_layout()

plt.show()



out = np.column_stack([
    t_grid,
    survival_w
])

np.savetxt(
    os.path.join(
        root,
        'Statistics/res_time/weighted_survival.txt'
    ),
    out,
    header='Residence_time(ps) Weighted_survival_probability',
    fmt='%.8e'
)