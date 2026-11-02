import matplotlib.pyplot as plt
import torch
import numpy as np
from ase.io import read
from math import sqrt, pi
from ase.geometry import find_mic
from Tools.Structure_calc.basic import  base_expansion
import os
from scipy.interpolate import griddata
from matplotlib import pyplot as plt

root = r"/mnt/e/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter6/Production/4site_5ns/"
def create_Gaussian_electron_density():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import matplotlib
    matplotlib.use('TkAgg')

    in_ = os.path.join(root,"Final_traj_for_DFT.traj")

    frames = read(in_, index=':')

    print(f"Total frames: {len(frames)}")

    # 设定原子参数
    valence_electrons = {'H': 1, 'O': 6}
    atomic_radii = {'H': 0.32, 'O': 0.66}

    # 构建网格（基于第一帧）
    cell = frames[0].get_cell()
    box = frames[0].cell.cellpar()[:3]
    grid_points = [int(x / 0.1) for x in box]
    nx, ny, nz = grid_points

    x = np.linspace(0, 1, nx, endpoint=False)
    y = np.linspace(0, 1, ny, endpoint=False)
    z = np.linspace(0, 1, nz, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_frac = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    grid_cart = grid_frac @ cell  # shape (N, 3)
    grid_cart = torch.tensor(grid_cart, dtype=torch.float32, device=device)


    # 函数定义
    def g(dist2, A, width):
        coef = A / (width ** 3 * (2 * pi) ** 1.5)
        return coef * torch.exp(-dist2 / (2 * width ** 2))

    # 初始化累加密度
    density_sum = torch.zeros(len(grid_cart), dtype=torch.float32, device=device)

    n_frames = 0
    # 遍历帧
    for i, atoms in enumerate(frames):
        select = []
        for idx, ele in enumerate(atoms.get_chemical_symbols()):
            if ele in ['C', 'N', 'Fe']:
                select.append(idx)

        pos, ele = atoms.get_positions(wrap=True), atoms.get_chemical_symbols()  # pos: <class 'numpy.ndarray'>  ele: <class 'list'>

        x_base, y_base, z_base = base_expansion(['C', 'N', 'Fe'], pos, ele, box)

        # 目标原子的坐标
        mask_target = np.isin(ele, ['O', 'H'])
        target_coords = pos[mask_target, :]
        target_ele = np.array(ele)[mask_target]
        x_target, y_target, z_target = target_coords[:, 0], target_coords[:, 1], target_coords[:, 2]
        # 对每个target原子计算其“相对 z”
        z_surface = griddata((x_base, y_base), z_base, (x_target, y_target), method='cubic')  # 拟合曲面（可以改为更精细的插值方式）
        z_relative = z_target - z_surface

        # 构造高斯参数
        gaussians = []
        for x,y,z,symbol in zip( x_target, y_target, z_relative, target_ele):
            assert symbol in valence_electrons
            atom_position = np.array([x,y,z%box[2]])
            A = valence_electrons[symbol]
            width = atomic_radii[symbol]
            gaussians.append((torch.tensor(atom_position, dtype=torch.float32, device=device), A, width))

        # 当前帧密度
        density = torch.zeros(len(grid_cart), dtype=torch.float32, device=device)

        # 累加原子的高斯贡献
        for center, A, width in gaussians:
            if center[2]<3 and A==6:
                print(center[2],A,width)
            delta = grid_cart - center
            for d in range(3):
                L = box[d]
                delta[:, d] -= L * torch.round(delta[:, d] / L)
            dist2 = (delta ** 2).sum(dim=1)
            density += g(dist2, A, width)

        density_sum += density
        n_frames +=1

        # # 对每个target原子计算其“相对 z”
        # z_all= griddata((x_base, y_base), z_base, (pos[:,0], pos[:,1]), method='cubic')  # 拟合曲面（可以改为更精细的插值方式）
        # pos[:,2] -= z_all
        # atoms.set_positions(pos)
        # atoms.wrap()

        print(f"Frame {i + 1}/{len(frames)} done.")

    density_avg = (density_sum / n_frames).reshape(nx, ny, nz).cpu().numpy()

    out = os.path.join(root,'Final_traj_for_DFT__avg_gaussian_electron_density__acc_0.1A.cube')
    # out = os.path.join(root,'test.cube')
    from ase.io.cube import  write_cube
    from ase.io import write
    with open(out,mode='w') as f:
        write_cube(f,atoms,data = density_avg) #atoms只是随机的一个背景。
    #
    # out = os.path.join(root,'test.traj')
    #
    # write(out,atoms,format = 'traj')
    # write(os.path.join(root,'test.xyz'),atoms)

def read_gaussian_electron_density():
    import matplotlib
    matplotlib.use('TkAgg')

    in_ = os.path.join(root,'Final_traj_for_DFT__avg_gaussian_electron_density__acc_0.1A.cube')
    from ase.io.cube import read_cube
    with open(in_,mode='r') as f:
        data = read_cube(f)

    atoms = data['atoms']
    box = atoms.cell.cellpar()[:3]
    Lz = box[2]
    grid_points = [int(x / 0.1) for x in box]
    nx, ny, nz = grid_points
    density_avg = data['data']


    # 计算 z 平均（shape: nz）
    rho_z= density_avg.mean(axis=(0, 1))  # shape: (nz,)
    # 生成对应的 z 坐标
    z = np.linspace(0, box[2], nz, endpoint=False)  # 实际 z 值（非分数坐标）

    rho_bulk = np.array([r for x,r in zip(z,rho_z) if 16<=x<=25]).mean() #0.26668614927879625
    rho_bulk  = 0.26668614927879625
    print(z,rho_bulk)

    data = [((z + Lz / 2) % Lz - Lz / 2, rho) for z, rho in zip(z, rho_z)]
    data.sort()
    data=np.array(data)

    z = data[:,0]
    rho_z = data[:,1]

    ideal_water_eps = 1.0 + 0.7 * rho_z / rho_bulk
    plt.plot(z,ideal_water_eps)
    plt.show()


    out = r"/mnt/e/Project_MNC4/inv_eps_calc_ok/ideal_water_eps.txt"
    with open(out, "w") as f:
        f.write("z(Ang)  eps\n")
        for zz, v in zip(z, ideal_water_eps):
            f.write(f"{zz:15.4f} {v:15.6f}\n")


# create_Gaussian_electron_density()
read_gaussian_electron_density()