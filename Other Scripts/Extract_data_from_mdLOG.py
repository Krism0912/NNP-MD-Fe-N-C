import sys
import  numpy as np
from matplotlib import pyplot as plt
import os
import matplotlib
matplotlib.use('TkAgg')

root = r"/mnt/e/Project_MNC4/NNP/Fe_NC/Fe_NC_all/FeNC-300H2O/MD_RUN_1"
in_ = os.path.join(root,'md.log')
out = os.path.join(root,'md_data.txt')
f=open(in_,mode='r')
out = open(out,mode='w')
out.write('{}     {}     {}     {}      {}\n'.format('Step','Epot','Temp','f_sd','max_f_sd'))
for _ in range(4):
    f.readline()
step=[]
epot=[]
temp=[]
f_sd=[]
max_f_sd=[]
for line in f.readlines():
    s=line.split()
    t = 21 if s[3][0]=='W' else 20
    out.write('{}     {}     {}     {}      {}\n'.format(s[6],s[8], s[12], s[18],s[t]))
    step.append(int(s[6]))
    epot.append(float(s[8]))
    temp.append(float(s[12]))
    f_sd.append(float(s[18]))
    max_f_sd.append(float(s[t]))

out.close()

plt.plot(step,epot,temp)
plt.show()