#!/bin/bash
# Job name:
#SBATCH --job-name=scan2cad_run_4_25_2020
#
# Account:
#SBATCH --account=fc_vivelab
#
# Partition:
#SBATCH --partition=savio2_gpu
#
# Quality of Service:
#SBATCH --qos=qos_name
#
# Wall clock limit:
#SBATCH --time=00:10:00
#
## Command(s) to run:
module load python
module load cuda/10.1
cd /global/scratch/akashgokul
conda activate /global/scratch/akashgokul/kaolin_run
cd kaolin/examples/Classification
