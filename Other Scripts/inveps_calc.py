import os
import sys
from itertools import pairwise

import scipy
import numpy as np
from scipy import constants, special
from matplotlib import pyplot as plt
import time
import seaborn as sns

import matplotlib

matplotlib.use('TkAgg')

EPSILON = constants.epsilon_0 / constants.elementary_charge * constants.angstrom
print(EPSILON)

from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# 自定义渐变 colormap
cdict1 = {
    'red': (
        (0.0, 0.0, 0.0),   # 蓝端无红
        (0.5, 0.5, 0.5),   # 灰色中间
        (1.0, 1.0, 1.0),   # 红端全红
    ),
    'green': (
        (0.0, 0.25, 0.25), # 蓝端带少量绿
        (0.5, 0.5, 0.5),   # 中间灰
        (1.0, 0.0, 0.0),   # 红端略带绿，使红更亮（偏橙红）
    ),
    'blue': (
        (0.0, 1.0, 1.0),   # 蓝端
        (0.5, 0.5, 0.5),   # 灰
        (1.0, 0.0, 0.0),   # 红端无蓝
    ),
    'alpha': (
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    )
}

blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)

# 用 ScalarMappable 模拟颜色映射
norm = Normalize(vmin=-0.25, vmax=0.25)
sm = ScalarMappable(norm=norm, cmap=blue_red1)

sm.set_array([])



def DFT_converged(root):
    if not os.path.exists(root):
        return False
    outcar = os.path.join(root, 'OUTCAR')
    with open(outcar, 'r') as f:
        while line:=f.readline():
            if "aborting loop EDIFF was not reached (unconverged)" in line:
                return False
    return True

def read_data(root,filename):
    data_x, data_y = [], []
    with open(os.path.join(root, filename)) as f:
        while line := f.readline():
            try:
                x, y = map(float, line.split())
                data_x.append(x)
                data_y.append(y)
            except:
                pass
    return data_x,data_y



def get_half_data(conv,idx,left):
    index = 0 if left else 1
    if choose[idx][index]=='+':
        E1,E2 = conv[-1]
    elif choose[idx][index]=='-':
        E1,E2 = conv[0]
    else:
        E1,E2 = round(choose[idx][index]-0.05,2),choose[idx][index]
        E1 = 0 if int(E1) ==E1 else E1
        E2 = 0 if int(E2) == E2 else E2

    dE = E2 - E1

    data_x, data_y = read_data(os.path.join(root,f"E={E1}/idx = {idx}"),
                               "Charge_z_curve_base.txt")
    _, data_y2 = read_data(os.path.join(root,f"E={E2}/idx = {idx}"),
                           "Charge_z_curve_base.txt")

    delta_rho = [y2 - y1 for y1, y2 in zip(data_y, data_y2)]

    data = [((z + Lz / 2) % Lz - Lz / 2, -y) for z, y in zip(data_x, delta_rho)]  # 注意负号
    data.sort()

    xs = np.array([z for z, _ in data])
    ys = np.array([rho for _, rho in data])

    int_curve = scipy.integrate.cumulative_trapezoid(ys, xs, initial=0)

    xs_new = xs
    ys_new = int_curve

    ys_new = [y / EPSILON / dE for y in ys_new]
    ys_new = np.array([1.0 - y for y in ys_new])
    mask = np.where(xs_new<0) if left == True else np.where(xs_new>=0)
    xs_new = xs_new[mask]
    ys_new = ys_new[mask]
    return xs_new,ys_new , E1


def draw_choosed_eps(root,idx):
    save_dir = os.path.join(root,'picture_eps_choose')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    out_x = np.array([])
    out_eps = np.array([])

    E_field = [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]

    conv = []
    for E1, E2 in pairwise(E_field):
        if not DFT_converged(os.path.join(root,f"E={E1}/idx = {idx}")):
            continue
        if not DFT_converged(os.path.join(root,f"E={E2}/idx = {idx}")):
            continue
        conv.append((E1, E2))

    print(idx, conv)
    if choose[idx]:
        xs_left, ys_left, E_left= get_half_data(conv, idx, True)
        xs_right, ys_right, E_right = get_half_data(conv, idx, False)
        xs = np.concatenate((xs_left, xs_right))
        ys = np.concatenate((ys_left, ys_right))
        plt.plot(xs, ys, label=f"{E_left}<->{E_right}")
        out_x = xs
        out_eps = 1.0 / ys

    xmax = 20
    plt.xlim(-xmax, xmax)
    plt.ylim(-0.1, 1.1)
    plt.xlabel(r'$z(\mathrm{\AA})$', fontsize=15)
    plt.ylabel(r'$ \epsilon^{-1} $', fontsize=15)
    plt.grid()
    plt.axhline(y=0.5555, color='r', linestyle='--')
    plt.axhline(y=0.588, color='r', linestyle='--')
    plt.axhline(y=1.0, color='grey', linestyle='-', linewidth=2)
    plt.axhline(y=0.0, color='grey', linestyle='-', linewidth=2)
    plt.title(f'idx = {idx}', fontsize=18)
    plt.legend(loc='best', fontsize=12)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(os.path.join(save_dir, 'idx={}.png'.format(idx)))
    plt.close()

    return out_x, out_eps


def draw_all_plot(root,idx):
    save_dir = os.path.join(root,'picture_eps')
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    out_x = None
    out_eps = None

    cbar = fig.colorbar(sm,
                        ax=axes.ravel().tolist(),  # 适配所有子图
                        orientation='vertical',
                        fraction=0.05,  # colorbar宽度比例
                        pad=0.03)  # 与子图间距

    cbar.set_label('EFIELD (V/Å)', fontsize=15)




    E_field = [-0.25,-0.2,-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15,0.2,0.25]

    # Draw V_Hatree
    for E in E_field:
        if not DFT_converged(os.path.join(root,f"E={E}/idx = {idx}")):
            print(idx,E)
            continue
        data_x, data_y = read_data(os.path.join(root,f"E={E}/idx = {idx}"),
                                   "V_Hatree_z_curve_base.txt")

        data = [((z + Lz / 2) % Lz - Lz / 2,y) for z,y in zip(data_x,data_y)]
        data.sort()
        xs = np.array([z for z,_ in data])
        ys = np.array([V for _,V in data])
        plt.subplot(2, 2, 1)
        plt.plot(xs, ys, label="E = {}V".format(E),color = sm.to_rgba(E/0.25))


    # Draw delta Ez
    conv =[]
    for E1, E2 in pairwise(E_field):

        if not DFT_converged(os.path.join(root,f"E={E1}/idx = {idx}")):
            continue
        if not DFT_converged(os.path.join(root,f"E={E2}/idx = {idx}")):
            continue
        conv.append((E1,E2))
        data_x, data_y = read_data(os.path.join(root,f"E={E1}/idx = {idx}"),
                                   "V_Hatree_z_curve_base.txt")
        _, data_y2 = read_data(os.path.join(root,f"E={E2}/idx = {idx}"),
                                   "V_Hatree_z_curve_base.txt")

        delta_V = [y2-y1 for y1,y2 in zip(data_y,data_y2)]

        data=[((z + Lz / 2) % Lz - Lz / 2,y) for z,y in zip(data_x,delta_V)]

        data.sort()

        xs = np.array([z for z,_ in data])
        ys = np.array([V for _,V in data])

        dz = xs[1] - xs[0]  # 均匀网格间距
        delta_Ez = -np.gradient(ys, dz)

        plt.subplot(2, 2, 3)
        plt.plot(xs, delta_Ez, label=r"$ E_{{{}}} - E_{{{}}}$".format(E2, E1),color = sm.to_rgba(E1/0.25))




    #Draw Delta rho and inv_eps
    for E1, E2 in pairwise(E_field):
        if not DFT_converged(os.path.join(root,f"E={E1}/idx = {idx}")):
            continue
        if not DFT_converged(os.path.join(root,f"E={E2}/idx = {idx}")):
            continue
        dE = E2 - E1
        data_x, data_y = read_data(os.path.join(root,f"E={E1}/idx = {idx}"),
                                   "Charge_z_curve_base.txt")
        _, data_y2 = read_data(os.path.join(root,f"E={E2}/idx = {idx}"),
                                   "Charge_z_curve_base.txt")

        delta_rho = [y2-y1 for y1,y2 in zip(data_y,data_y2)]

        data=[((z + Lz / 2) % Lz - Lz / 2,-y) for z,y in zip(data_x,delta_rho)] #注意负号
        data.sort()

        xs = np.array([z for z,_ in data])
        ys = np.array([rho for _,rho in data])

        plt.subplot(2,2,2)
        plt.plot(xs, ys, label=r"$\rho_{{{}}} - \rho_{{{}}}$".format(E2, E1),color = sm.to_rgba(E1/0.25))

        int_curve = scipy.integrate.cumulative_trapezoid(ys, xs, initial=0)

        # 如果你想要 x 起点从 -20，可以在前面补一个点：
        xs_new = xs #np.insert(xs, 0, -20)
        ys_new = int_curve #np.insert(int_curve, 0, 0)

        ys_new = [y/EPSILON/dE for y in ys_new]
        ys_new= np.array([1.0-y for y in ys_new])
        plt.subplot(2, 2, 4)
        plt.plot(xs_new, ys_new, label=r"$\rho_{{{}}} - \rho_{{{}}}$".format(E2, E1))

        out_x = xs_new
        out_eps = 1.0/ys_new


    xmax = 20
    plt.rcParams.update({'font.size': 16})
    plt.tick_params(labelsize=14)
    plt.subplot(2,2,1)
    plt.xlim(-xmax, xmax)
    plt.xlabel(r'$z(\mathrm{\AA})$',fontsize = 15)
    plt.ylabel(r'$V_{Hatree}$(V)',fontsize = 15)
    # plt.legend(loc='best')
    plt.grid()

    plt.subplot(2,2,2)
    plt.xlabel(r'$z(\mathrm{\AA})$',fontsize = 18)
    plt.ylabel(r'$\Delta \rho (\mathrm{e}/\mathrm{\AA}^3)$',fontsize = 15)
    plt.xlim(-xmax, xmax)
    plt.ylim(-10e-4, 10e-4)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))  # 设置y轴刻度标签为科学记数法
    # plt.legend(loc='best')
    plt.grid()

    plt.subplot(2,2,3)
    # plt.legend(loc='best')
    plt.xlabel(r'$z(\mathrm{\AA})$',fontsize = 15)
    plt.ylabel(r'$ \Delta E_{z} $(V)',fontsize = 15)
    plt.ylim(-0.1,0.1)
    plt.xlim(-xmax, xmax)
    plt.grid()

    plt.subplot(2,2,4)
    plt.xlim(-xmax, xmax)
    plt.ylim(-0.1, 1.1)
    plt.xlabel(r'$z(\mathrm{\AA})$',fontsize = 15)
    plt.ylabel(r'$ \epsilon^{-1} $',fontsize = 15)

    plt.grid()
    plt.axhline(y=0.5555,color='r',linestyle='--')
    plt.axhline(y=0.588,color='r',linestyle='--')
    plt.axhline(y=1.0,color='grey',linestyle='-',linewidth = 1.5)
    plt.axhline(y=0.0, color='grey', linestyle='-',linewidth = 1.5)
    plt.suptitle(f'idx = {idx}',fontsize = 18)
    plt.legend(loc='best',fontsize = 8)
    # 调整子图以避免覆盖
    plt.subplots_adjust(hspace = 0.25,wspace = 0.3, right=0.8)

    plt.savefig(os.path.join(save_dir, 'idx={}.png'.format(idx)),dpi=800,bbox_inches='tight')
    # plt.show()
    plt.close()


    return out_x,out_eps

from collections import  defaultdict
mark=[None,['+','+'],['-','-'],['+','-'],['-','+']]
choose = defaultdict(lambda :None)
with open(r"/mnt/e/Project_MNC4/inv_eps_calc_ok/eps_calc/choose.txt") as f:
    while line := f.readline():
        line = line.split()
        print(line)
        idx = int(line[0])
        if len(line)<2:
            continue
        elif line[1] in ['1','2','3','4']:
            choose[idx] = mark[int(line[1])]
        else:
            choose[idx] = [float(line[1]), float(line[2])]



Lz = 50  # 注意这个！！！！！！！！！！！！！！！！
total = []
root = "/mnt/e/Project_MNC4/inv_eps_calc_ok/eps_calc/"
for idx in range(51):
    draw_all_plot(root,idx)
    continue
    # z,out_eps = draw_choosed_eps(root,idx)
    # if out_eps.any():
    #     total.append(out_eps)

sys.exit(0)
total=np.array(total)
eps = total.mean(axis = 0)
print(eps)

import pandas as pd
m, n = total.shape

# 转为长格式 DataFrame
df = pd.DataFrame({
    "x": np.tile(z, m),
    "y": total.flatten(),
    "group": np.repeat(np.arange(m), n)
})

# 画平均曲线和误差带（默认置信区间95%，你也可以设置 ci='sd' 使用标准差）
ax = sns.lineplot(data=df, x="x", y="y",errorbar='sd')  # 或 ci=95（默认），或 ci=None 去掉误差带
plt.xlabel(r'$z(\mathrm{\AA})$', fontsize=15)
plt.ylabel(r'$ \epsilon$', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title(f'Average $\epsilon$ profile of FeNC-H2O system', fontsize=18)
ax.figure.set_size_inches(12,8)
plt.grid()
plt.xlim(-20,20)
plt.ylim(0.8,4)

plt.savefig(r"/mnt/e/Project_MNC4/inv_eps_calc_ok/Average_eps_profile.png")
plt.close()

out = r"/mnt/e/Project_MNC4/inv_eps_calc_ok/FeNC_eps.txt"
with open(out, "w") as f:
    f.write("z(Ang)  eps\n")
    for zz, v in zip(z, eps):
        f.write(f"{zz:15.4f} {v:15.6f}\n")

