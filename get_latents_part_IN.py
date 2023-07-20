import torch
import tqdm
import os
import re
import click
import pickle
import numpy as np
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from training import nirvana_utils
from torchvision.datasets import ImageNet, ImageFolder
from torch.utils.data import  Dataset, DataLoader
from torchvision import transforms



def noising(net, img, class_labels, num_steps=256, sigma_min=0.002, sigma_max=80, rho=7, device='cuda'):
    batch_size = img.shape[0]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    denoiseds = []
    xs = []

    # Main sampling loop.
    x_next = (img * 2 - 1).to(torch.float64)
    xs.append((x_next*0.5+0.5).clone())
    denoiseds.append((x_next*0.5+0.5).clone())
    
    for i, (t_cur, t_next) in tqdm.tqdm(list(reversed(list(enumerate(zip(t_steps[1:], t_steps[:-1]))))), unit='step'): # 0, ..., N-1
        x_cur = x_next

        # Euler step.
        t_used = t_cur if i < num_steps - 1 else torch.tensor(sigma_min, device=device)
        denoised = net(x_cur, t_used, class_labels).to(torch.float64)
        denoiseds.append((denoised * 0.5 + 0.5))
        d_cur = (x_cur - denoised) / t_used
        x_next = x_cur + (t_next - t_used) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_cur + (t_next - t_used) * (0.5 * d_cur + 0.5 * d_prime)
        xs.append((x_next*0.5+0.5).clone())

    return torch.stack(denoiseds).transpose(0, 1), torch.stack(xs).transpose(0, 1), x_next


#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--subdirs_class',           help='Create subdirectory for every class',                              is_flag=True)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--path_to_data',     help='path to image folder', metavar='DIR',                                     type=str, default=None)
@click.option('--path_to_label_distr',     help='Label distr to sample from the model', metavar='DIR',              type=str, default=None)

def main(network_pkl, outdir, subdirs_class, max_batch_size, path_to_data, path_to_label_distr=None, device=torch.device('cuda')):
    dist.init()

    dataset = ImageNet(path_to_data)

    tr = transforms.Compose([
        transforms.Resize(64), 
        transforms.CenterCrop(64), 
        transforms.ToTensor()])

    def collate_fn(x):
        imgs = []
        classes = []
        for img, cl in x:
            imgs.append(tr(img))
            classes.append(cl)
        return torch.stack(imgs), torch.tensor(classes, dtype=int)

    data_loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=max_batch_size, shuffle=False)

    # num_batches = ((len(dataset) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    # all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    # rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

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
    
    len_data_loader = (len(dataset) + max_batch_size - 1) // max_batch_size
    steps_by_each = len_data_loader // dist.get_world_size()

    # Loop over batches.
    dist.print0("Finding latents for IN")
    for i, batch in tqdm.tqdm(enumerate(data_loader), unit='batch', disable=(dist.get_rank() != 0)):
        
        batch_size = len(batch[0])
        if batch_size == 0:
            continue

        if i % dist.get_world_size() != dist.get_rank():
            continue
    
        if i < steps_by_each * dist.get_world_size():
            torch.distributed.barrier()

        # Pick latents and labels.

        imgs, labels = batch
        # latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = torch.eye(net.label_dim, device=device)[labels]

        _, _, noise = noising(net, imgs.to(device), class_labels)

        idxs = range(max_batch_size * i, max_batch_size * i + batch_size)

        for idx, noise_el, label in zip(idxs, noise, labels):
            if subdirs_class:
                latent_dir = os.path.join(outdir, f'{label}') 
            else:
                latent_dir = outdir

            os.makedirs(latent_dir, exist_ok=True)
            latent_path = os.path.join(latent_dir, f'{idx}.pth')
            torch.save(noise_el, latent_path)

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
