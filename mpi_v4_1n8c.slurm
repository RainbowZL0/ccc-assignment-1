#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00

module purge
module load spartan
module load foss/2022a
module load Python/3.10.4
module load SciPy-bundle/2022.05

export PYTHONPATH="$(pwd):$PYTHONPATH"
mpirun -n 8 python a004_assignment_1/a001_ndjson.py --version 4
