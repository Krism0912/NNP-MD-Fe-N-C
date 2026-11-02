#!/bin/bash
#SBATCH -J water_MTD
#SBATCH -p gpu
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

export PLUMED_KERNEL=/share/home/krism0912/plumed-2.10.0/src/lib/libplumedKernel.so
source activate py39
date
echo $CUDA_VISIBLE_DEVICES
python md_run_PLUMED.py
date
