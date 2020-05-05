#!/bin/bash
# Job name:
#SBATCH --job-name=scan2cad_transf_may3
#
# Account:
#SBATCH --account=fc_vivelab
#
# Partition:
#SBATCH --partition=savio2
#
# Request one node:
#SBATCH --nodes=1
#
# Request cores (24, for example)
#SBATCH --ntasks-per-node=2
#
#Request CPU
#SBATCH --cpus-per-task=8
#
# Wall clock limit:
#SBATCH --time=72:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akashgokul@berkeley.edu
## Command(s) to run (example):
echo "256 bz 100 ep "
echo "Run number inv_tf4_more_data"
module load python
# module load cuda/10.1
source activate /global/scratch/akashgokul/kaolin_run
python3 pointcloud_classification_scan2cad.py --epochs 100 --run_number transf4_128 --batch-size 128
