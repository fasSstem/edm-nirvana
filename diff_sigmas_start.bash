#!/bin/bash
#SBATCH --job-name=p2v-cls-mo
#SBATCH --no-requeue
#SBATCH --time=2-00:00
#SBATCH --begin=now
#SBATCH --signal=TERM@120
#SBATCH --output=slurm_logs/%j_%n_%x.txt

set -e
mkdir sigmas_dir

for i in 0 1 2 3 4 5 6 7 8 9 10 11
do
torchrun --standalone --nproc_per_node=8 generate_class_adaptive.py --outdir=fid-tmp-$i --steps=8 --batch=50 --seeds=0-49999 --subdirs --network=pretrained/edm-imagenet-64x64-cond-adm.pkl --sigmas=sigmas_tuned_equal/$i.pt

echo "run_number_$i"
# Calculate FID
torchrun --standalone --nproc_per_node=8 fid.py calc --images=fid-tmp-$i --ref=imagenet-64x64.npz
done