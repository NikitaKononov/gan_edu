from __future__ import print_function
import yaml
# %matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from clearml import Task

from model import Generator, Discriminator

criterion = nn.BCELoss()
real_label = 1.
fake_label = 0.


def train(netG, netD, optimizerG, optimizerD, device, dataloader, fixed_noise, config):
    logger = Task.current_task().get_logger()
    global_step = 0
    for epoch in range(config['num_epochs']):
        for i, data in enumerate(dataloader, 0):
            # Discriminator: real
            netD.zero_grad()

            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1)

            errD_real = criterion(output, label)

            errD_real.backward()
            D_x = output.mean().item()

            # Discriminator: fake
            noise = torch.randn(b_size, config['nz'], 1, 1, device=device)

            fake = netG(noise)
            label.fill_(fake_label)

            output = netD(fake.detach()).view(-1)

            errD_fake = criterion(output, label)

            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake

            optimizerD.step()

            # Generator
            netG.zero_grad()
            label.fill_(real_label)

            output = netD(fake).view(-1)

            errG = criterion(output, label)

            errG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()

            if global_step % config['log_interval'] == 0:
                print('Global step: %d, Epoch: %d/%d, Step: %d/%d\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): '
                      '%.4f / %.4f'
                      % (global_step, epoch, config['num_epochs'], i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                logger.report_scalar(title="Unified graph", series="Loss_D", iteration=global_step, value=errD.item())
                logger.report_scalar(title="Unified graph", series="Loss_G", iteration=global_step, value=errG.item())
                logger.report_scalar(title="Unified graph", series="D(x)", iteration=global_step, value=D_x)
                logger.report_scalar(title="Unified graph", series="D_G_z1", iteration=global_step, value=D_G_z1)
                logger.report_scalar(title="Unified graph", series="D_G_z2", iteration=global_step, value=D_G_z2)

            if global_step % config['eval_interval'] == 0 or (
                    (epoch == config['num_epochs'] - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()

                logger.report_image(
                    "image",
                    "image PIL",
                    iteration=global_step,
                    image=np.transpose(vutils.make_grid(fake, padding=2, normalize=True).detach().cpu().numpy(),
                                       (1, 2, 0))
                )
            global_step += 1


def read_config():
    with open("config.yaml", "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def init_training():
    config = read_config()

    task = Task.init(project_name=config['proj_name'], task_name=config['exp_name'] + ' ' + config['exp_ver'])
    task.connect_configuration(configuration=config)

    print("Seed: ", config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    dataset = dset.ImageFolder(root=config['dataroot'],
                               transform=transforms.Compose([
                                   transforms.Resize(config['image_size']),
                                   transforms.CenterCrop(config['image_size']),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'],
                                             shuffle=True, num_workers=config['workers'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = Generator().to(device)
    netD = Discriminator().to(device)

    print('Generator total trainable params {}'.format(sum(p.numel() for p in netG.parameters() if p.requires_grad)))
    print('Discriminator total trainable params {}'.format(sum(p.numel() for p in netD.parameters() if p.requires_grad)))

    # print(netG)
    # print(netD)

    fixed_noise = torch.randn(64, config['nz'], 1, 1, device=device)

    optimizerG = optim.Adam(netG.parameters(), lr=float(config['lr_g']), betas=(config['beta1'], 0.999))
    optimizerD = optim.AdamW(netD.parameters(), lr=float(config['lr_d']), betas=(config['beta1'], 0.999))

    train(netG, netD, optimizerG, optimizerD, device, dataloader, fixed_noise, config)


if __name__ == '__main__':
    read_config()
    init_training()
