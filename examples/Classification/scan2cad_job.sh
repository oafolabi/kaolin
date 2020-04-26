#!/bin/bash
# Job name:
#SBATCH --job-name=scan2cad_apr_25_2020
#
# Account:
#SBATCH --account=fc_vivelab
#
# Partition:
#SBATCH --partition=savio2_gpu
#
# Request one node:
#SBATCH --nodes=1
#
# Request cores (24, for example)
#SBATCH --ntasks-per-node=2
#
#Request GPUs
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
## Command(s) to run (example):
module load python
module load cuda/10.1
source activate /global/scratch/akashgokul/kaolin_run
python3 pointcloud_classification_scan2cad.py --run_number 'Apr26_1_64_20' --tensorboard 0 --batch_size 64 --epochs 20