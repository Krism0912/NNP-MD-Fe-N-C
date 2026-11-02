#!/bin/bash
#SBATCH -o 2.out
#SBATCH -e 1.err
#SBATCH -J DFT1
#SBATCH -p amd_256
#SBATCH -n 128
#SBATCH -N  2
source activate py39
date
python DFT_calc.py
date
