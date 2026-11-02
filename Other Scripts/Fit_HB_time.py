import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

roo_cutoff = 3.5
angle_cutoff=  30
dt = 0.05
threshold = 0

#Read Data
root = r"/mnt/e/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter6/Production/4site_5ns/"


out_dir = os.path.join(root,'Statistics/HB_data')
out = open(os.path.join(out_dir,'Fit_result_double.txt'),mode='w')
# 表头
out.write(f"{'R_cutoff(A)':>15}\t{'angle_cutoff(°)':>15}\t{'dt(ps)':>8}\t{'threshold':>10}\t{'area':>10}\t"
          f"{'A':>10}\t{'B':>10}\t{'C':>10}\t{'D':>10}\t{'R²':>10}\t{'τ1(ps)':>15}\t{'τ2(ps)':>10}\n")


def fit():
    input_root = os.path.join(root, 'Statistics/HB_data/Roo = {}A angle = {}°'.format(roo_cutoff, angle_cutoff))
    in_dir = os.path.join(input_root, 'dt={}_threshold={}').format(dt, threshold)

    x_data=[]
    y_data=[]
    with open(os.path.join(in_dir,f'{tag[area]}_hb_curve_block_1.txt')) as f:
        f.readline()
        while s:= f.readline():
            x,y=map(float,s.split())
            x_data.append(x)
            y_data.append(y)


    # # 示例数据
    # x = np.linspace(0, 10, 100)
    # # 假设真实参数为 A=2, b=1.5, C=1, d=0.5
    # y = 2 * np.exp(-1.5 * x) + 1 * np.exp(-0.5 * x) + 0.1 * np.random.normal(size=len(x))  # 加点噪声

    x = np.array(x_data)
    y = np.array(y_data)
    # 定义拟合函数
    def double_exp(x, A, b, C, d):
        return A * np.exp(-b * x) + C * np.exp(-d * x)

    bounds = ([0, 1e-4, 0, 1e-4], [1.5, np.inf, 1.5, np.inf])
    try:
        popt, pcov = curve_fit(double_exp, x, y, p0=(1, 1, 1, 0.1),bounds = bounds, maxfev = 10000)  # p0 是初始猜测
    except:
        print('error')
        out.write('error\n')
        return

    # 输出拟合参数
    A, B, C, D = popt
    print(f"A = {A}, b = {B}, C = {C}, d = {D}")
    from basic import fit_tau_double_exp
    tau_fast, tau_slow, params, r2 = fit_tau_double_exp(x, y)
    A,b,C,d = params
    print(f"A = {A}, b = {B}, C = {C}, d = {D}")
    # 绘图比较拟合效果
    plt.rcParams.update({'font.size': 14})
    plt.figure(dpi=600)
    plt.scatter(x, y, label='data', s=10)
    plt.plot(x, double_exp(x, *popt), 'r-', label='fit: A=%.2f, b=%.2f, C=%.2f, d=%.2f' % tuple(popt))
    plt.legend()
    plt.xlabel('Time(ps)')
    plt.ylabel('C(t)')
    plt.title('Double Exponential Fit')
    plt.show()


    # 获取拟合结果
    y_pred = double_exp(x, *popt)

    # 计算 R^2
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R^2 = {r_squared:.4f}")


    # 数据行
    out.write(f"{roo_cutoff:15.2f}\t{angle_cutoff:15.0f}\t{dt:8.2f}\t{threshold:10.0f}\t{tag[area]}\t"
              f"{A:10.2f}\t{B:10.2f}\t{C:10.2f}\t{D:10.2f}\t{r_squared:10.3f}\t"
              f"{1/min(B, D):15.3f}\t{1/max(B, D):10.3f}\n")

min_dt = 0.05
tag ={0:'adsorb',1:'1st layer',2:'2nd layer',3:'bulk'}
roo_cutoff = 3.5
angle_cutoff = 30
area = 0
dt = 0.1
threshold = 5
fit()
sys.exit()
import time
for roo_cutoff in [3.0,3.2,3.5]:
    for angle_cutoff in [20,30]:
        for area in range(4):
            for interval in [1,2,5,10]:
                dt = min_dt * interval
                for threshold in [0,1,2,5]:
                    fit()
                    sys.exit(0)
                    # time.sleep(1)
