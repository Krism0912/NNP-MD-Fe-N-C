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


# Load the entire trajectory
# root = r"/mnt/e/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter4/Production/150H2O-box"
root = r"/mnt/e/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter6/Production/4site_5ns/"
in_=os.path.join(root,"MD_RUN_1/MD_for_analysis.traj")

frames=Trajectory(in_)

traj_size = len(frames)
box = frames[0].cell.cellpar()[:3] # NVT simulation box

waters = identify_water_molecules(frames[0])

data = []
o_idxs = np.array([triplet[0] for triplet in waters])
h1_idxs = np.array([triplet[1] for triplet in waters])
h2_idxs = np.array([triplet[2] for triplet in waters])

h_idxs = np.append(h1_idxs,h2_idxs)


tag ={0:'adsorb',1:'1st layer',2:'2nd layer',3:'bulk'}

def HB_statistics(start,end=len(frames),gap=1):
    import json
    print('loading data...')
    input_data = os.path.join(root, 'Statistics/HB_data/Roo = 3.5A angle = 30°/hbond_data.json')

    with open(input_data, "r") as f:
        load_dict = json.load(f)
    h_bond_data = defaultdict(set)
    for k,v in load_dict.items():
        h_bond_data[int(k)] = set([tuple(x) for x in v])

    # input_data =  os.path.join(root, 'Statistics/H_atom_type.json')
    # with open(input_data, "r") as f:
    #     layer1,layer2,bulk = json.load(f)
    #
    # h_data = [defaultdict(set) for _ in range(3)]
    #
    # for k,v in layer1.items():
    #     h_data[0][int(k)] = set([x for x in v])
    # for k,v in layer2.items():
    #     h_data[1][int(k)] = set([x for x in v])
    # for k,v in bulk.items():
    #     h_data[2][int(k)] = set([x for x in v])

    input_data = os.path.join(root, 'Statistics/water_area_type_new.json')
    with open(input_data, "r") as f:
        adsorb_O,layer1_O,layer2_O,bulk_O = json.load(f)

    o_data = [defaultdict(set) for _ in range(4)]
    for k,v in adsorb_O.items():
        o_data[0][int(k)] = set([x for x in v])
    for k,v in layer1_O.items():
        o_data[1][int(k)] = set([x for x in v])
    for k,v in layer2_O.items():
        o_data[2][int(k)] = set([x for x in v])
    for k,v in bulk_O.items():
        o_data[3][int(k)] = set([x for x in v])

    print('Complete!')

    pos = [None] * end
    for i in range(start,end,gap):
        pos[i] = frames[i].get_positions(wrap=True)
    O_type_count= [[0,0] for _ in range(4)]
    cnt= [defaultdict(int) for _ in range(4)]

    keys = set()
    data = [[] for _ in range(4)]
    area = 0
    for idx in range(start,end,gap):
        print(idx,'size=',len(h_bond_data[idx]))
        water_bond = [defaultdict(lambda:None) for _ in range(4)]
        for j in range(4):
            for o in o_data[j][idx]:
                water_bond[j][o]=[0,0]  #第j个区域，某个O原子为donor和acceptor的计数

        for bond in h_bond_data[idx]:
            h,o_d,o_a = bond
            for i,o in enumerate([o_d,o_a]):
                for j in range(4):
                    if o in o_data[j][idx]:
                        O_type_count[j][i] += 1
                        water_bond[j][o][i] += 1
                        break
        for j in range(4):
            for _,v in water_bond[j].items():
                cnt[j][tuple(v)] += 1
                keys.add(tuple(v))

        for area in range(4):
            for bond in h_bond_data[idx]:
                h,o_d,o_a = bond
                krism = int(o_d in o_data[area][idx]) + int(o_a in o_data[area][idx])  #判断HB在某个区域的标准
                if (area==0 and krism==1)  or krism==2:  # 更改
                    d_oo = pbc_distance(pos[idx][o_d],pos[idx][o_a],box)
                    d_oh1 = pbc_distance(pos[idx][h],pos[idx][o_d],box)
                    d_oh2 = pbc_distance(pos[idx][h],pos[idx][o_a],box)
                    theta_hoo = pbc_angle(pos[idx][h],pos[idx][o_d],pos[idx][o_a],box)
                    theta_oho = pbc_angle(pos[idx][o_d],pos[idx][h],pos[idx][o_a],box)
                    data[area].append([d_oo,d_oh1,d_oh2,theta_hoo,theta_oho])    # 更改这里即可

    # data = np.array(data)
    # x = data[:, 1]-data[:, 2]
    # y = data[:, 3]
    for area in range(4):
        d = np.array(data[area])
        # x = d[:, 0]
        # y = d[:, 3]
        x = d[:, 1]-d[:, 2]
        y = d[:, 3]
        from scipy.ndimage import gaussian_filter

        bins = 200
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)

        # 可选：高斯模糊处理（增强视觉效果）
        heatmap = gaussian_filter(heatmap, sigma=1.5)


        out_dir = os.path.join(root,'Statistics/HB_statistics')
        os.makedirs(out_dir,exist_ok=True)

        # 画图
        plt.imshow(
            heatmap.T, origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap='jet', aspect='auto'
        )
        plt.colorbar(label='Density')
        plt.title(r'Hydrogen Bond statistics ($d_{OO}$ vs $theta$)'+f'_{tag[area]}') #修改
        plt.xlabel('$d_{OO}$ (Å)')
        plt.ylabel(r'$theta$ ')
        plt.tight_layout()
        plt.show()



        out = open(os.path.join(out_dir, f'HB_geometry_statistics_{tag[area]}.txt'), mode='w') ##### 修改
        out.write('Start frame: {} End frame: {}  gap: {}\n'.format(start,end, gap))
        out.write('d_OO(Ang) d_OH1(Ang)  d_OH2(Ang) theta_HOO(°) theta_OHO(°)\n')
        for d_oo,d_oh1,d_oh2,theta_hoo,theta_oho in data[area]:
            out.write('{}   {}   {}   {}  {}\n'.format(d_oo,d_oh1,d_oh2,theta_hoo,theta_oho))
        out.close()

        out = open(os.path.join(out_dir, f'HB_O_cnt_{tag[area]}.txt'), mode='w')  ##### 修改

        out.write('Area  O_d     O_a \n')
        for type,(x,y) in zip(["adsorb_O","layer1_O","layer2_O","bulk_O"],O_type_count):
            out.write('{}    {}    {}\n'.format(type,x,y))
        out.close()

        out = open(os.path.join(out_dir, 'data_cnt.txt'), mode='w') ##### 修改
        out.write('Start frame: {} End frame: {}  gap: {}\n'.format(start,end, gap))
        keys = list(keys)
        keys.sort()
        out.write('Area  ' + '   '.join(str(x) for x in keys)+'\n')

        for type,datas in zip(["adsorb_O","layer1_O","layer2_O","bulk_O"],cnt):
            out.write('{}    '.format(type))
            for k in keys:
                out.write('{}  '.format(datas[k]))
            out.write('\n')

        mx_donor =  max(k[0] for k in keys)
        out.write('Donor\n')
        out.write('Area  ' + '   '.join([str(i) for i in range(mx_donor + 1)]) + '\n')
        for type,datas in zip(["adsorb_O","layer1_O","layer2_O","bulk_O"],cnt):
            tmp = [0] * (1 + mx_donor)
            out.write('{}    '.format(type))
            for k in keys:
                tmp[k[0]] += datas[k]
            for v in range(mx_donor+1):
                out.write('{}  '.format(tmp[v]))
            out.write('\n')

        mx_acceptor =  max(k[1] for k in keys)
        out.write('Acceptor\n')
        out.write('Area  ' + '   '.join([str(i) for i in range(mx_acceptor + 1)]) + '\n')
        for type,datas in zip(["adsorb_O","layer1_O","layer2_O","bulk_O"],cnt):
            tmp = [0] * (1 + mx_acceptor)
            out.write('{}    '.format(type))
            for k in keys:
                tmp[k[1]] += datas[k]
            for v in range(mx_acceptor+1):
                out.write('{}  '.format(tmp[v]))
            out.write('\n')
        out.close()



def save_data_for_H():
    up1 = 4.82
    up2 = 7.65
    bulk_down,bulk_up = 18,24

    layer1 = defaultdict(list)
    layer2 = defaultdict(list)
    bulk = defaultdict(list)
    # adsorb = defaultdict(list)
    z_box = box[2]

    import json
    print('loading data...')
    input_data = os.path.join(root, 'Statistics/water_area_type.json')
    with open(input_data, "r") as f:
        zmin_data,_,_,_,_ = json.load(f)
    print('zmin_data load complete!')
    print(len(zmin_data))


    for i,frame in enumerate(frames):
        print('frame:',i)
        pos = frame.get_positions(wrap=True)
        # ele = frame.get_chemical_symbols()

        h_pos = pos[h_idxs]
        z_min = zmin_data[i]

        for idx,pos in zip(h_idxs,h_pos):
            z = pos[2]
            dis = min(abs(z-z_min),abs(z - (z_min+z_box) ))
            if dis<up1:
                layer1[i].append(int(idx))
            elif up1<=dis<up2:
                layer2[i].append(int(idx))
            elif bulk_down<=dis<=bulk_up:
                bulk[i].append(int(idx))

    out_data = [layer1,layer2,bulk]

    import json
    out_dir =  os.path.join(root, 'Statistics')
    out = os.path.join(out_dir, 'H_atom_type.json')
    os.makedirs(out_dir, exist_ok=True)
    with open(out, "w") as f:
        f.write(json.dumps(out_data))



# for area in range(4):
HB_statistics(start=0,end=len(frames),gap = 1)
# save_data_for_H()