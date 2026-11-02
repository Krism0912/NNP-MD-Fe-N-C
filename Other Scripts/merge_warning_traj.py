import  os.path
from ase.io import read, write, Trajectory
root=r"/mnt/e/Project_MNC4/NNP/Fe_NC/Fe_NC_multi_site/Iter2"
out_traj=[]
folders=os.listdir(root)
folders.sort(key=lambda x:(len(x),x))
print(folders)
source=[]
for folder in ['MD_300K/MD_RUN_1','MD_300K/MD_RUN_2','MD_300K_2/MD_RUN_1']:
    path = os.path.join(root, folder)
    print(path)
    try:
        warning_traj=Trajectory(os.path.join(root,folder,'warning_struct.traj'))
        out_traj += warning_traj
        source += len(warning_traj)*[folder]
    except:
        pass
print(len(out_traj))
warning_traj=write(os.path.join(root,'total_warning_struct.traj'),out_traj)
write(os.path.join(root, 'total_warning_struct.xyz'),out_traj)
for i,source_name in enumerate(source):
    print(i,source_name)
