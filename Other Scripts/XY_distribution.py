import numpy as np
import matplotlib.pyplot as plt
from ase.io import read,Trajectory
from scipy.ndimage import gaussian_filter
import os
from basic import  get_slab,check_in_region
from collections import  defaultdict
root = r"/mnt/e/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter6/Production/4site_5ns/"
in_=os.path.join(root,"MD_RUN_1/MD_for_analysis.traj")


traj=Trajectory(in_)

n = len(traj)


x_axis,y_axis,z_axis = traj[0].cell.cellpar()[:3] # NVT simulation box

# 提取所有帧中 O 原子的 x, y 坐标
positions_up = []
positions_down = []
positions = []


limit = True
slab = True
low = 18
high = 24
both_side = False

o_idxs = np.array([idx for idx,ele in enumerate(traj[0].get_chemical_symbols()) if ele=='O'])
Fe_idxs = np.array([idx for idx,ele in enumerate(traj[0].get_chemical_symbols()) if ele=='Fe'] )



import json
print('loading data...')
input_data = os.path.join(root, 'Statistics/water_area_type_new_with_basedata.json')
with open(input_data, "r") as f:
    _,adsorb_O, layer1_O, layer2_O, bulk_O = json.load(f)

o_data = [defaultdict(set) for _ in range(4)]
for k, v in adsorb_O.items():
    o_data[0][int(k)] = set([x for x in v])
for k, v in layer1_O.items():
    o_data[1][int(k)] = set([x for x in v])
for k, v in layer2_O.items():
    o_data[2][int(k)] = set([x for x in v])
for k, v in bulk_O.items():
    o_data[3][int(k)] = set([x for x in v])
print('complete!')
tag ={0:'adsorb',1:'1st layer',2:'2nd layer',3:'bulk'}

def f(start, end, gap):
    n_frames = 0
    for i in range(start, min(n, end), gap):
        n_frames += 1
        pos = traj[i].get_positions(wrap=True)

        # z_min = get_slab(pos,ele,['C','N','Fe'])
        ref_pos = pos[Fe_idxs[0]][:2] - [x_axis/4,y_axis/4]
        print('frame',i)
        for o_idx in o_data[area][i]:
            x,y,z = pos[o_idx]
            # if check_in_region(z,z_min,low,high,both_side,z_axis):
            o_pos = np.array([x, y]) - ref_pos
            o_pos[0] = o_pos[0] % x_axis
            o_pos[0] -= x_axis if o_pos[0] > x_axis / 2 else 0
            o_pos[1] = o_pos[1] % y_axis
            o_pos[1] -= y_axis if o_pos[1] > y_axis / 2 else 0
            positions.append(o_pos)
    return n_frames


up1 = 4.82
up2 = 7.82
bulk_down = 18
bulk_up = 24

Lz = [None, 4.82 * 2,3 * 2,6 ] #因为1st ,2ndlayer有对称的两侧部分，需要乘以2
for area in range(3,4):
    if area in [0,2,3]:
        positions = []
    n_frames = f(0,n,10)
    if area == 0:
        #draw adsorb and 1st_layer together
        continue

    positions = np.array(positions)
    x = positions[:, 0]
    y = positions[:, 1]

    # 设置 2D 网格（可以根据 slab 尺寸调整）
    bins = 250
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)

    # 可选：高斯模糊处理（增强视觉效果）
    heatmap = gaussian_filter(heatmap, sigma=1.5)
    plt.rcParams.update({'font.size': 14})
    plt.figure(dpi=600)

    NA = 6.02214076e23
    m_H2O = 18.01528 / NA  # g

    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]

    V_bin_cm3 = dx * dy * Lz[area] * 1e-24

    density = heatmap * m_H2O / (n_frames * V_bin_cm3)

    # 画图
    plt.imshow(
        density.T, origin='lower',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='jet', aspect='auto'
    )
    plt.colorbar(label='Water density (g/cm$^3$)')
    plt.title(f'2D Distribution of O Atoms in {tag[area]}')
    plt.xlabel('X (Å)')
    plt.ylabel('Y (Å)')
    plt.tight_layout()

    from collections import  defaultdict
    element_coords = defaultdict(list)  # key: element symbol, value: list of positions

    slab_atoms = traj[0]

    for atom in slab_atoms:
        if atom.symbol == 'Fe':
            ref_pos = atom.position[:2]-[x_axis/4,y_axis/4]
            break
    for atom in slab_atoms:
        if atom.symbol in ['Fe','N','C']:
            pos = atom.position[:2]-ref_pos
            pos[0] = pos[0] % x_axis
            pos[0] -= x_axis if pos[0] > x_axis / 2 else 0
            pos[1] = pos[1] % y_axis
            pos[1] -= y_axis if pos[1] > y_axis / 2 else 0
            element_coords[atom.symbol].append(pos)




    element_style = {
        'Fe': {'color': 'red', 'size': 80},
        'N': {'color': 'aquamarine', 'size': 50},
        'C': {'color': 'yellow', 'size': 40},
        # 你可以按需添加其他元素
    }

    for elem, positions in element_coords.items():
        positions = np.array(positions)
        style = element_style.get(elem, {'color': 'green', 'size': 30})  # 默认样式
        if elem == 'Fe':
            continue
        else:
            plt.scatter(
                positions[:, 0], positions[:, 1],
                c=style['color'],
                s=style['size'],
                edgecolors='black',
                label=elem
            )
    plt.axis('equal')
    plt.show()