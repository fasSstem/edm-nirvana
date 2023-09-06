import torch
torch.set_num_threads(16)

from torch import nn
from torch.utils.data import  Dataset, DataLoader
import torchvision.transforms as transforms
# import pytorch_lightning as pl
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
import tqdm
from copy import deepcopy
import pickle
import dnnlib
from training import nirvana_utils
import lpips

import click

def step(net, t_cur, t_next, x_cur, i, num_steps, class_labels):
    # Euler step.
    denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
    d_cur = (x_cur - denoised) / t_cur
    x_next = x_cur + (t_next - t_cur) * d_cur

    # Apply 2nd order correction.
    if i < num_steps - 1:
        denoised = net(x_next, t_next, class_labels).to(torch.float64)
        d_prime = (x_next - denoised) / t_next
        x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next

def denoising(net, latents, class_labels, t_steps, sigma_first = torch.tensor(80.0, device='cuda'), sigma_last = torch.tensor(0.0, device='cuda')):
    batch_size = latents.shape[0]
    
    num_steps = len(t_steps) + 1

    # Time step discretization.

    # Main sampling loop.
    x_next = latents.to(torch.float64) * sigma_first
    i = 0
    x_next = step(net, sigma_first, t_steps[0], x_next, i, num_steps, class_labels)
    for i, (t_cur, t_next) in list(enumerate(zip(t_steps[:-1], t_steps[1:]))): # 0, ..., N-1
        x_next = step(net, t_cur, t_next, x_next, i+1, num_steps, class_labels)
    x_next = step(net, t_steps[-1], sigma_last, x_next, num_steps - 1, num_steps, class_labels)
    return x_next

def get_sigmas(model, num_steps, sigma_min=0.002, sigma_max=80, rho=7):
    sigma_min = max(sigma_min, model.sigma_min)
    sigma_max = min(sigma_max, model.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device='cuda')
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    return t_steps[1:-1]

class CustomDataset(Dataset):
    def __init__(self, path, latents):
        self.path = path
        self.latents = latents
        self.pics = sorted([a for a in os.listdir(self.path) if 'png' in a])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: 2 * x - 1
        ])
    
    def __getitem__(self, i):
        
        return self.transform(Image.open(self.path+self.pics[i])), self.pics[i], self.latents[i]
    
    def __len__(self):
        return len(self.pics)


def change_grad_sigmas(sigmas):
    pass
    for i in range(4):
        sigmas.grad[i] *= 5 ** (4 - i)
    #sigmas_10._grad[0] = 0
    #sigmas_10._grad[-1] = 0
    #print(sigmas._grad[-1], sigmas[-1])
    if sigmas._grad[-1] > 0 and sigmas[-1] < 0.1:
        sigmas._grad[-1] = 0
    #print(sigmas._grad[-1])
    
def change_sigmas(sigmas):
    if sigmas[-1] < 2e-3:
        sigmas[-1] = 2e-3




@click.command()
@click.option('--index',                   help='Index of process', metavar='INT',                                  type=click.IntRange(min=0), required=True)
@click.option('--outdir',                  help='Where to save sigmas', metavar='DIR',                              type=str, required=True)
@click.option('--path_alex',               help='path to alexnet', metavar='DIR',                                   type=str, required=True)
@click.option('--is_alex',                 help='Use alex in loss',                                                 is_flag=True)
@click.option('--path_imagenet',           help='path to imagenet dir', metavar='DIR',                              type=str, required=True)
@click.option('--path_sigmas_start',       help='path to sigmas start', metavar='DIR',                              type=str, required=True)
@click.option('--path_edm',                help='path to edm', metavar='DIR',                                       type=str, required=True)
@click.option('--epochs',                  help='num of epochs', metavar='INT',                                     type=click.IntRange(min=0), required=True)
@click.option('--lr',                      help='learning_rate', metavar='FLOAT',                                   type=float, default=1e-2, show_default=True)

def main(index, outdir, path_alex, is_alex, path_imagenet, path_sigmas_start, path_edm, epochs, lr):
    device = 'cuda'
    with open(path_edm, 'rb') as f:
        model = pickle.load(f)['ema'].to(device)
        
    model.eval()
    latents_1000 = []
    for name in sorted([a for a in os.listdir(path_imagenet) if 'pt' in a]):
        tens = torch.load(path_imagenet+name)
        latents_1000.append(tens)
    latents_1000 = torch.cat(latents_1000)
    latents_1000.shape

    sigmas_start = torch.load(path_sigmas_start)[0].detach()

    sigmas_10_all = torch.zeros(125, 8, 7)
    loss_fn_alex = lpips.LPIPS(net=('alex' if is_alex else 'vgg'), model_path=path_alex, pnet_rand=True) # best forward scores
    loss_fn_alex.cuda()
    loss_fn_alex = loss_fn_alex.double()
    bs = 10
    for cl in range(125 * index, 125 * (index + 1)):
        print(cl)
        sigmas_10 = sigmas_start.clone().cuda()
        sigmas_10.requires_grad = True
        optim = torch.optim.Adam([sigmas_10], lr=lr)
        dataset = CustomDataset(path_imagenet+f'{cl:04}/', latents_1000)
        data = DataLoader(dataset, shuffle=True, batch_size=bs, num_workers=8)
        classes = torch.ones(size=[bs], device=device, dtype=int) * cl
        class_labels = torch.eye(model.label_dim, device=device)[classes]
        for epoch in range(epochs):
            for i, batch in tqdm.tqdm(enumerate(data), total=100):
                ref_imgs, _, latents = batch
                img_gen = denoising(model, latents.cuda(), class_labels, sigmas_10)
                loss = loss_fn_alex(img_gen, ref_imgs.cuda()).mean() # loss_func(img_10_cur, img_256)
                optim.zero_grad()
                loss.backward()
                change_grad_sigmas(sigmas_10)
                optim.step()
                if i % 50 == 49:
                    sigmas_10_all[cl - 125 * index, i // 50 + epoch * 2] = sigmas_10
        torch.save(sigmas_10_all, f'{outdir}/sigmas_classes_{index}.pt')
    
    nirvana_utils.copy_out_to_snapshot(outdir)



if __name__ == "__main__":
    main()