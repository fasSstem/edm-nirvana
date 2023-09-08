#!/bin/bash
#SBATCH --job-name=p2v-cls-mo
#SBATCH --no-requeue
#SBATCH --time=2-00:00
#SBATCH --begin=now
#SBATCH --signal=TERM@120
#SBATCH --output=slurm_logs/%j_%n_%x.txt

set -e

for (( i=0; i<=$2; i++))
do
torchrun --standalone --nproc_per_node=8 generate_class_adaptive.py --outdir=fid-tmp-$i --steps=8 --batch=50 --seeds=0-49999 --subdirs --network=pretrained/edm-imagenet-64x64-cond-adm.pkl --sigmas=$1/$i.pt

echo "run_number_$i"
# Calculate FID
torchrun --standalone --nproc_per_node=8 fid.py calc --images=fid-tmp-$i --ref=imagenet-64x64.npz
done