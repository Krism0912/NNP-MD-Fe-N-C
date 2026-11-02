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
root = r"/mnt/e/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter6/Production/4site_5ns/"
# in_=os.path.join(root,"MD_RUN_1/MD_for_analysis.traj")
in_=os.path.join(root,"MD_RUN_1/MD.traj")


frames=Trajectory(in_)
frames = frames[-50000:]


traj_size = len(frames)
box = frames[0].cell.cellpar()[:3] # NVT simulation box


dt = 0.05         # time per frame (ps), adjust if known
max_lag = int(100/dt)    # number of frames for correlation

slab = True
low = 0
high = 5
volume = box[0]*box[1]*(high-low)


waters = identify_water_molecules(frames[0])
Fe_idxs = np.array([atom.index for atom in frames[0] if atom.symbol == 'Fe'])
data = []
o_idxs = np.array([triplet[0] for triplet in waters])
h1_idxs = np.array([triplet[1] for triplet in waters])
h2_idxs = np.array([triplet[2] for triplet in waters])
tag ={0:'adsorb',1:'1st layer',2:'2nd layer',3:'bulk'}

def adsorb_water():
    import json
    print('loading data...')
    input_data = os.path.join(root, 'Statistics/Density_Profile/water_area_type.json')
    with open(input_data, "r") as f:
        adsorb,_,_,_ = json.load(f)

    input_data = os.path.join(root, 'Statistics/side_data.json')
    with open(input_data, "r") as f:
        side_data = json.load(f)

    # in_ = os.path.join(root, "MD_RUN_1/MD_for_analysis.traj")
    # frames = Trajectory(in_)
    print('complete!')

    idxs=[[] for _ in range(4)]
    side = [[] for _ in range(4)]
    up_d = [[] for _ in range(4)]
    down_d = [[] for _ in range(4)]
    range_ =list(range(0,len(frames)))
    for i in range_:
        print(i,len(range_))
        i = str(i)
        for j in range(4):
            adsorb_idx = adsorb[i][j]
            idxs[j].append(adsorb_idx)
            side[j].append(int(side_data[i][j][-1]))

            up_d[j].append( pbc_distance( np.array(side_data[i][j][4]), np.array(side_data[i][j][3]),box) )
            down_d[j].append(pbc_distance( np.array(side_data[i][j][5]), np.array(side_data[i][j][3]),box) )

    time_axis = np.array(range_)*dt


    # def smooth(y, window=10):
    #     y_pad = np.pad(y, (window // 2, window - 1 - window // 2), mode='reflect')
    #     y_smooth = np.convolve(y_pad, np.ones(window) / window, mode='valid')
    #     return y_smooth
    # up_d[0] = smooth(up_d[0])
    # down_d[0] = smooth(down_d[0])
    plt.rcParams.update({'font.size': 16})
    for i in range(4):
        fig, ax = plt.subplots(dpi = 600)
        plt.plot(time_axis,up_d[i],label = f'site{i+1}-up',alpha = 0.5)
        plt.plot(time_axis, down_d[i], label=f'site{i+1}-down',alpha = 0.5,linestyle = '-')
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Fe-water distance($\AA$)")

        ax.set_title(f"Minimum Fe-water distance(site{i+1})")
        plt.legend(loc = 'best')
        plt.tight_layout()
        plt.savefig(os.path.join(root, f'Statistics/res_time/Minimum Fe-H2O distance_site{i+1}.jpg'), dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
    return


def adsorb_water_index():
    Fe_cnt = len(Fe_idxs)
    idxs=[[] for _ in range(Fe_cnt)]
    range_ =list(range(0,len(frames)))

    for i in range_:
        print(i)
        pos, ele = frames[i].get_positions(wrap=True), frames[i].get_chemical_symbols()

        o_pos = pos[o_idxs]
        # Extract the adsorbed water
        for j,fe_idx in enumerate(Fe_idxs):
            fe_pos = pos[fe_idx]
            delta = o_pos - fe_pos
            # Apply minimum image convention
            delta -= box * np.round(delta / box)
            # Compute distances
            dists = np.linalg.norm(delta, axis=1)
            min_idx = np.argmin(dists)
            target_o = o_idxs[min_idx]
            idxs[j].append(int(target_o) if np.min(dists) <= 2.6 else np.nan)  # 注意，这里保存的是原来的idx，不是0-319


    time_axis = np.array(range_)*dt

    plt.figure(dpi = 600)
    plt.rcParams.update({'font.size': 14})
    for i in range(Fe_cnt):
        plt.scatter(time_axis,idxs[i],label = f'site{i+1}',s=10)
    plt.xlabel("Time (ps)")
    plt.ylabel("Water index")
    plt.title(f"Index of adsorbed water")
    plt.legend(loc = 'upper left')
    plt.tight_layout()
    if not os.path.exists(os.path.join(root, f'Statistics/res_time')):
        os.makedirs(os.path.join(root, f'Statistics/res_time'))
    plt.savefig(os.path.join(root, f'Statistics/res_time/water_index.jpg'), dpi=600,
                bbox_inches='tight')
    plt.show()
    plt.close()

    np.save(os.path.join(root, 'Statistics/res_time/idxs.npy'), np.array(idxs))
    return



def fit_tau_double_exp(x, y):
    from scipy.optimize import curve_fit

    def double_exp(x, A, b, C, d):
        return A * np.exp(-b * x) + C * np.exp(-d * x)

    # 初值（很关键，影响收敛）
    p0 = (0.5, 1.0, 0.5, 0.1)

    # 限制参数：全部为正
    bounds = (0, np.inf)

    popt, _ = curve_fit(double_exp, x, y, p0=p0, bounds=bounds, maxfev=10000)

    A, b, C, d = popt

    # τ（解析积分）
    tau = A / b + C / d

    # R^2
    y_pred = double_exp(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return tau, (A, b, C, d), r2


def fit_tau_single_exp(x, y):
    from scipy.optimize import curve_fit
    import numpy as np

    def single_exp(x, A, b):
        return A * np.exp(-b * x)

    # 初值（可以稍微调一下更稳）
    p0 = (1.0, 0.1)

    # 限制参数为正
    bounds = (0, np.inf)

    popt, _ = curve_fit(single_exp, x, y, p0=p0, bounds=bounds, maxfev=10000)

    A, b = popt

    # τ（解析积分）
    tau = 1 / b

    # R^2
    y_pred = single_exp(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return tau, (A, b), r2
def residence_time(start,end = len(frames)):
    import json
    print('loading data...')
    input_data = os.path.join(root, 'Statistics/Density_Profile/water_area_type.json')
    with open(input_data, "r") as f:
        adsorb_O, layer1_O, layer2_O, bulk_O = json.load(f)

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

    out_dir = os.path.join(root, 'Statistics/res_time/threshold_{}/'.format(threshold * 0.05))
    os.makedirs(out_dir, exist_ok=True)

    range_ = list(range(start,end-max_lag,1))

    block_size = 10000  # 自己设
    blocks = [range_[i:i + block_size:100] for i in range(0, len(range_), block_size)]

    time_axis = np.arange(max_lag) * dt
    block_tau_all = [[] for _ in range(4)]

    total_correlation = np.array([np.zeros(max_lag) for _ in range(4)])
    for b_idx, block in enumerate(blocks):
        print(f"\nProcessing block {b_idx} ...")

        block_corr_total = np.array([np.zeros(max_lag) for _ in range(4)])
        frame_cnt = 0


        for start_frame in block:
            frame_cnt += 1
            corr = np.array([np.zeros(max_lag) for _ in range(4)])

            for area in range(4):
                cnt = 0
                for idx in o_data[area][start_frame]:
                    cnt += 1
                    pre_t = -1
                    for i in range(max_lag):
                        if idx in o_data[area][start_frame + i]:
                            corr[area][i] += 1
                            pre_t = i
                        elif i - pre_t > threshold and threshold >= 0: # threshold <0 means no check
                            break

                if cnt > 0:
                    corr[area] /= cnt

                block_corr_total[area] += corr[area]
        block_corr_total /= frame_cnt

        total_correlation += block_corr_total

        for area in range(4):
            tau_block, params, r2 = fit_tau_single_exp(time_axis, block_corr_total[area])
            block_tau_all[area].append(tau_block) #这个block 的 tau

            # ===== 文件名带 block index =====
            fname_prefix = f"{tag[area]}_block_{b_idx}"

            A, b = params
            textstr = (
                f"A = {A:.3f}\n"
                f"b = {b:.3f}\n"
                f"$R^2$ = {r2:.3f}"
            )
            # ---- 曲线图 ----
            plt.plot(time_axis, block_corr_total[area])
            plt.xlabel("Time (ps)")
            plt.ylabel(r"$r_{H2O}(t)$")
            plt.title(f"{tag[area]} Block {b_idx}")
            plt.grid()
            plt.text(
                0.65, 0.95, textstr,  # 位置（相对坐标）
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )
            plt.savefig(os.path.join(out_dir, f"{fname_prefix}_curve.jpg"))
            plt.close()
            print(f"{tag[area]} Block {b_idx} τ = {tau_block:.3f} ps")


    ## 所有的block的数据的tau
    output_file = os.path.join(out_dir, "final_tau_all.txt")

    with open(output_file, "w") as out:
        out.write("Area tau SD\n")

        for area in range(4):
            tau_array = np.array(block_tau_all[area])

            tau_mean = np.mean(tau_array)

            if len(tau_array) > 1:
                tau_std = np.std(tau_array, ddof=1)
            else:
                tau_std = 0.0

            # 写成一行
            out.write(f"{tag[area]} {tau_mean:.2f} {tau_std:.2f}\n")

    # 所有block的平均曲线
    total_correlation /= len(blocks)

    for area in range(4):
        avg_corr = total_correlation[area]
        tau_avg = np.trapz(avg_corr, time_axis)
        print(f"{tag[area]} Avg curve tau = {tau_avg:.3f} ps")
        plt.plot(time_axis, avg_corr)
        plt.xlabel("Time (ps)")
        plt.ylabel(r"$r_{H2O}(t)$")
        plt.title(f"{tag[area]} Average Correlation (All Blocks)")
        plt.grid()
        plt.savefig(os.path.join(out_dir, f"{tag[area]}_avg_curve.jpg"))
        plt.close()
        with open(os.path.join(out_dir, f"{tag[area]}_avg_curve.txt"), "w") as out:
            out.write("time correlation\n")
            for t, c in zip(time_axis, avg_corr):
                out.write(f"{t} {c}\n")



# adsorb_water()
adsorb_water_index()
# for threshold in [-1,0,1,2,5,10]:
#     residence_time(threshold)