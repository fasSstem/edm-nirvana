#!/bin/bash
#SBATCH --job-name=p2v-cls-mo
#SBATCH --no-requeue
#SBATCH --time=2-00:00
#SBATCH --begin=now
#SBATCH --signal=TERM@120
#SBATCH --output=slurm_logs/%j_%n_%x.txt

set -e
mkdir sigmas_dir

for i in 0 1 2 3 4 5 6 7
CUDA_VISIBLE_DEVICES=$i python -m tuning_sigmas --index=$i --outdir=sigmas_dir --path_alex=alex.pt --path_imagenet= --path_sigmas_start= --path_edm= &
done
wait

python -m concat_sigmas