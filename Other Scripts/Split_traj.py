import json,os
import math
import numpy as np
from ase.io import read, write, Trajectory
from pathlib import Path
import sys

root=r"/mnt/e/Project_MNC4/Dataset/NVT/multi-site/Iter7_DFT_Bilayer"
for traj_file in Path(root).rglob('DFT_SP_calc.traj'):
    print(traj_file)
    dataset=Trajectory(traj_file)

    n=len(dataset)
    test_ratio=0.1
    num_test = int(math.ceil(n * test_ratio))
    indices = np.random.permutation(n)
    splits = {
        "train and validation": indices[num_test:].tolist(),
        "test": indices[:num_test].tolist(),
    }

    # Save split file
    with open(os.path.join(root, "datasplits.json"), "w") as f:
        json.dump(splits, f)

    train_val_traj=[]
    test_traj=[]
    for idx in splits["train and validation"]:
        train_val_traj.append(dataset[idx])
    for idx in splits['test']:
        test_traj.append(dataset[idx])

    write(os.path.join(root, "test_set.traj"), test_traj)
    write(os.path.join(root, "train_val_set.traj"), train_val_traj)