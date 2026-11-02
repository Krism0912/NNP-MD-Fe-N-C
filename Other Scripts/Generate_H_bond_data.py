import sys
from ase.io import read,write,Trajectory
import numpy as np
import math
import os

from basic import *
from collections import defaultdict

# Load the entire trajectory
root = r"/mnt/e/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter6/Production/150H2O_box_2ps"
in_=os.path.join(root,"MD_RUN_1/MD_for_analysis.traj")
frames=Trajectory(in_)

traj_size = len(frames)
print('traj_size',traj_size)
box = frames[0].cell.cellpar()[:3] # NVT simulation box


waters = identify_water_molecules(frames[0])

data = []
o_idxs = np.array([triplet[0] for triplet in waters])
h1_idxs = np.array([triplet[1] for triplet in waters])
h2_idxs = np.array([triplet[2] for triplet in waters])

def save_data():
    print('Parameters: Roo = {}  angle = {} '.format(roo_cutoff,angle_cutoff))
    # Saved HB data
    hbond_data =  defaultdict(list)

    for idx, frame in enumerate(frames):
        if idx%5==0:
            print('now', idx)
        pos = frame.get_positions(wrap=True)
        o_pos = pos[o_idxs]  # shape: (N, 3)

        # 计算所有 O-O 距离矩阵（带 PBC）
        diff = o_pos[:, np.newaxis, :] - o_pos[np.newaxis, :, :]  # shape: (N, N, 3)
        diff -= box * np.round(diff / box)  # PBC correction
        dist_mat = np.linalg.norm(diff, axis=-1)  # shape: (N, N)

        # 原始 (i,j) O-O 对索引
        i_idx, j_idx = np.where((dist_mat < roo_cutoff) & (dist_mat > 0))

        # 扩展为所有 H（每对两个）
        i_all = np.concatenate([i_idx, i_idx])
        j_all = np.concatenate([j_idx, j_idx])
        h_all = np.concatenate([h1_idxs[i_idx], h2_idxs[i_idx]])

        # 构造批量坐标
        h_pos = pos[h_all]  # H donor
        o_d_pos = o_pos[i_all]  # O donor
        o_a_pos = o_pos[j_all]  # O acceptor

        # 一次性计算所有角度
        angles = pbc_angle_vec(h_pos, o_d_pos, o_a_pos, box)

        # 筛选满足条件的索引
        mask = angles <= angle_cutoff


        # 提取结果（最多 N²×2 项）
        for h, od, oa in zip(h_all[mask], o_idxs[i_all][mask], o_idxs[j_all][mask]):
            hbond_data[idx].append((int(h), int(od), int(oa)))



    import json
    out_dir =  os.path.join(root, 'Statistics/HB_data/Roo = {}A angle = {}°'.format(roo_cutoff,angle_cutoff) )
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'hbond_data.json')
    with open(out, "w") as f:
        f.write(json.dumps(hbond_data))


for roo_cutoff in [3.5]:
    for angle_cutoff in [30]:
        save_data()

