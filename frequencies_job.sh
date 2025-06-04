#!/bin/bash
#SBATCH --account=def-girouxb1
#SBATCH --time=0-00:30           # time (DD-HH:MM)
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB    
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexi.morin@inrs.ca
#SBATCH --output=./outputs/slurm-%A_%a.out

source /home/alexim/projects/def-girouxb1/alexim/microseismic_dl/bin/activate
python frequencies.py --task-id $SLURM_ARRAY_TASK_ID

