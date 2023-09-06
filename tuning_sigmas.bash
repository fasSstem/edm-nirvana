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
do
CUDA_VISIBLE_DEVICES=$i python3 -m tuning_sigmas --index=$i --outdir=sigmas_dir --path_alex=$3 $2 --path_imagenet=gen_imagenet/ --path_sigmas_start=$5 --path_edm=pretrained/edm-imagenet-64x64-cond-adm.pkl --epochs=$1 --lr=$4 &
done
wait

python3 -m concat_sigmas --outdir=sigmas_dir