import torch

from torch import nn
from torch.utils.data import  Dataset, DataLoader

import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import click

import numpy as np
import os

import tqdm

from copy import deepcopy
from torchvision.datasets import ImageFolder
import pickle
from timm import create_model

def get_collate_fn(trans):
    
    def collate_fn(x):
        imgs = []
        classes = []
        for img, cl in x:
            imgs.append(trans(img))
            classes.append(cl)
        return torch.stack(imgs), torch.tensor(classes, dtype=int)
    
    return collate_fn



def get_precision_and_recall(real_embs, gen_embs, k):

    dists_real_gen = torch.cdist(real_embs.cuda(), gen_embs.cuda()).cpu()
    dists_real_real = torch.cdist(real_embs.cuda(), real_embs.cuda()).cpu()
    dists_gen_gen = torch.cdist(gen_embs.cuda(), gen_embs.cuda()).cpu()

    r_k_gen = dists_gen_gen.topk(k=k+1, dim=1, largest=False).values[:, k].cpu()
    r_k_real = dists_real_real.topk(k=k+1, dim=1, largest=False).values[:, k].cpu()

    print(f'recall = {((dists_real_gen < r_k_gen[None, :]).sum(dim=1) > 0).float().mean().item()}')
    print(f'precision = {((dists_real_gen < r_k_real[:, None]).sum(dim=0) > 0).float().mean().item()}')



@click.command()
@click.option('--network', 'network_path',  help='Network path', metavar='PATH|URL',                                type=str, required=True)
@click.option('--is_vgg',                  help='Is model vgg or dinov2',                                           is_flag=True)
@click.option('--real_dir',                help='Where are real images', metavar='DIR',                             type=str, required=True)
@click.option('--gen_dir',                 help='Where are gen images', metavar='DIR',                              type=str, required=True)
@click.option('--num_neigh',               help='Parameter k', metavar='DIR',                                       type=click.IntRange(min=1), default=3, show_default=True)


def main(network_path, is_vgg, real_dir, gen_dir, num_neigh):
    real = ImageFolder(real_dir)
    gen = ImageFolder(gen_dir)


    if is_vgg:
        with open(network_path, 'rb') as handle:
            model = pickle.load(handle)
        model.eval()
        model.cuda()
    else:
        model = create_model('vit_large_patch14_dinov2')
        model.load_state_dict(torch.load(network_path))
        model.cuda()
        model.eval()

    tr = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])

    real_dl = DataLoader(real, collate_fn=get_collate_fn(tr), batch_size=100, shuffle=False, num_workers=16)
    gen_dl = DataLoader(gen, collate_fn=get_collate_fn(tr), batch_size=100, shuffle=False, num_workers=16)

    real_embs = []
    for batch in tqdm.tqdm(real_dl):
        im, _ = batch
        with torch.no_grad():
            real_embs.append(model(im.cuda()).cpu())
    real_embs = torch.cat(real_embs)

    gen_embs = []
    for batch in tqdm.tqdm(gen_dl):
        im, _ = batch
        with torch.no_grad():
            gen_embs.append(model(im.cuda()).cpu())
    gen_embs = torch.cat(gen_embs)

    get_precision_and_recall(real_embs, gen_embs, num_neigh)



if __name__ == "__main__":
    main()