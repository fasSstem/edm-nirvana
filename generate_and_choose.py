# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from training import nirvana_utils
from timm import create_model
from torch import nn
from torchvision import transforms

from generate import edm_sampler, ablation_sampler, StackedRandomGenerator, parse_int_list

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--subdirs_class',           help='Create subdirectory for every class',                              is_flag=True)
@click.option('--wo_last',                 help='Stop at T=2e-3',                                                   is_flag=True)
@click.option('--use_realism_1',           help='Use first real_metric',                                            is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

@click.option('--path_to_label_distr',     help='Label distr to sample from the model', metavar='DIR',              type=str, default=None)

def main(network_pkl, outdir, subdirs, subdirs_class, wo_last, use_realism_1, seeds, class_idx, max_batch_size, path_to_label_distr=None, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    dif_ns = [256, 128, 40, 16, 8]

    beit = create_model('beitv2_large_patch16_224')
    network_path = 'beitv2l.pth'
    beit.head = nn.Identity()
    beit.load_state_dict(torch.load(network_path))
    beit.cuda()
    beit.eval()

    real_embs = torch.load('IN_val_embs.pth')
    dists_real_real = torch.cdist(real_embs.cuda(), real_embs.cuda()).cpu()
    k = 3
    r_k_real = dists_real_real.topk(k=k+1, dim=1, largest=False).values[:, k].cpu()


    tr_beit_tens = transforms.Compose([
    transforms.Resize(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

    def get_realism_score(imgs):
        with torch.no_grad():
            gen_embs = beit(tr_beit_tens(imgs).float())
            
        dists_real_gen = torch.cdist(real_embs.cuda(), gen_embs).cpu()

        realism_1 = (r_k_real[:, None] / dists_real_gen).max(dim=0).values

        realism_2 = dists_real_gen.topk(k=3, dim=0, largest=False).values.mean(dim=0)
        return realism_1, realism_2

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    all_num_steps = []

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        labels = None
        if net.label_dim:
            # TODO: add probs to randint according to dinov2 distr
            if path_to_label_distr:
                label_distr = torch.from_numpy(np.load(path_to_label_distr)).to(torch.float)
                labels = torch.multinomial(label_distr, batch_size, replacement=True)
            else:
                labels = rnd.randint(net.label_dim, size=[batch_size], device=device)
            class_labels = torch.eye(net.label_dim, device=device)[labels]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler

        images_ns = []
        realism_1, realism_2 = [], []
        for ns in dif_ns:
            images_el = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like, wo_last=wo_last, num_steps=ns, **sampler_kwargs)
            images_ns.append(images_el)
            realism_1_el, realism_2_el = get_realism_score(images_el)
            realism_1.append(realism_1_el)
            realism_2.append(realism_2_el)
        realism_1 = torch.stack(realism_1)
        realism_2 = torch.stack(realism_2)
        images_ns = torch.stack(images_ns)
        if use_realism_1:
            num = realism_1.argmax(dim=0)
        else:
            num = realism_2.argmin(dim=0)

        all_num_steps += [dif_ns[ind] for ind in num]

        images = images_ns[num, torch.arange(batch_size)]

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np, label in zip(batch_seeds, images_np, labels):
            if subdirs:
                image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') 
            elif subdirs_class:
                image_dir = os.path.join(outdir, f'{label}') 
            else:
                image_dir = outdir

            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)
    print(sum(all_num_steps) / len(all_num_steps))

    # Copy images to nirvana snapshot path
    if dist.get_rank() == 0:
        nirvana_utils.copy_out_to_snapshot(outdir)

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
