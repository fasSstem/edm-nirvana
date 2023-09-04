import click
import torch
import os
from training import nirvana_utils

@click.command()
@click.option('--outdir',                  help='Where to save sigmas', metavar='DIR',                              type=str, required=True)

def main(outdir):
    tens = []
    for file in sorted(os.listdir(outdir)):
        tens.append(torch.load(f'{outdir}/{file}'))
    tens = torch.cat(tens)
    torch.save(tens, f'{outdir}/sigmas_classes.pt')
    nirvana_utils.copy_out_to_snapshot(outdir)


if __name__ == "__main__":
    main()