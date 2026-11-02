import numpy as np
import os


root = r"/mnt/e/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter6/Production/4site_5ns"
tag ={0:'adsorb',1:'1st layer',2:'2nd layer',3:'bulk'}
tmp_root = os.path.join(root, 'Statistics/MSD_and_Diffusion')
data = np.load(os.path.join(tmp_root, 'msd_data_area_all.npz'))

time_axis = data['time_axis']

total_time = time_axis[-1]
print('total time:',total_time)
all_MSD_x_all  = data['all_MSD_x']   # (4, n_samples, max_lag)
all_MSD_y_all  = data['all_MSD_y']
all_MSD_z_all  = data['all_MSD_z']
all_MSD_xy_all = data['all_MSD_xy']
all_MSD_xyz_all = data['all_MSD_xyz']

tag ={0:'adsorb',1:'1st_layer',2:'2nd_layer',3:'bulk'}
label = ['x','y','z','xy','xyz']
dim = [1,1,1,2,3]
def fit_msd_slope(t, y, tmin= total_time * 0.5, tmax = total_time):
    """
    data : list of list/tuple, 格式 [[t, MSD_x, MSD_y, MSD_z, MSD_xy], ...]
    tmin, tmax : 拟合区间 (ps)
    col : 要拟合的列 (1=MSD_x, 2=MSD_y, 3=MSD_z, 4=MSD_xy)

    返回: (slope, intercept)
    """
    # 转 numpy 数组
    # 选取区间
    mask = (t >= tmin) & (t <= tmax)
    t_fit = t[mask]
    y_fit = y[mask]

    # 线性拟合 (polyfit 一次多项式)
    slope, intercept = np.polyfit(t_fit, y_fit, 1)
    return slope



out = open(os.path.join(tmp_root, 'diffusion_with_error.txt'), 'w')
out.write('System   Dx_mean  Dx_sem   Dy_mean  Dy_sem   Dz_mean  Dz_sem   Dxy_mean  Dxy_sem   Dxyz_mean Dxyz_sem\n')

for area in range(4):

    results = []

    datasets = [
        all_MSD_x_all[area],
        all_MSD_y_all[area],
        all_MSD_z_all[area],
        all_MSD_xy_all[area],
        all_MSD_xyz_all[area]
    ]

    for i, all_MSD in enumerate(datasets):

        Ds = []

        for msd in all_MSD:   # 每条 MSD 曲线

            if np.all(np.isnan(msd)):
                continue

            mask_time = (time_axis >= total_time * 0.2)
            t_sel = time_axis[mask_time]
            y_sel = msd[mask_time]

            valid = ~np.isnan(y_sel)

            if np.sum(valid) < 10:
                continue

            t_fit = t_sel[valid]
            y_fit = y_sel[valid]

            # 线性拟合
            slope, _ = np.polyfit(t_fit, y_fit, 1)

            dimension = dim[i]
            D = slope / (2.0 * dimension)

            Ds.append(D)

        Ds = np.array(Ds)

        if len(Ds) == 0:
            D_mean = np.nan
            D_sem = np.nan
        else:
            valid_D = ~np.isnan(Ds)

            if np.sum(valid_D) == 0:
                D_mean = np.nan
                D_sem = np.nan
            else:
                D_mean = np.nanmean(Ds)
                D_sem = np.nanstd(Ds) / np.sqrt(np.sum(valid_D))

        # 单位转换
        D_mean *= 10
        D_sem *= 10

        results.append((D_mean, D_sem))

        print(f"{tag[area]} | component {label[i]} → D = {D_mean:.3f} ± {D_sem:.3f}")

    out.write(
        f"{tag[area]}   "
        f"{results[0][0]:.3f} {results[0][1]:.3f}   "
        f"{results[1][0]:.3f} {results[1][1]:.3f}   "
        f"{results[2][0]:.3f} {results[2][1]:.3f}   "
        f"{results[3][0]:.3f} {results[3][1]:.3f}   "
        f"{results[4][0]:.3f} {results[4][1]:.3f}\n"
    )

out.close()