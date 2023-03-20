import argparse
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from model import VAE

parser = argparse.ArgumentParser(description='vae with convs training')

parser.add_argument('--batch_size', type=int, help='training batch size', default=256)
parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
parser.add_argument('--epochs', type=int, help='training epochs', default=25)
parser.add_argument('--eval_interval', type=int, help='eval after steps', default=1000)
parser.add_argument('--log_interval', type=int, help='log after steps', default=100)

args = parser.parse_args()


def eval(net, eval_loader, device, step):
    net.eval()
    running_loss = .0
    for idx, data in enumerate(eval_loader):
        imgs, _ = data
        imgs = imgs.to(device)

        out, mu, logVar = net(imgs)

        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence

        running_loss += loss

        save_num = randint(0, args.batch_size - 1)
        inp_save = np.transpose(imgs[save_num].cpu().detach().numpy(), [1, 2, 0])
        out_save = np.transpose(out[save_num].cpu().detach().numpy(), [1, 2, 0])

        plt.subplot(121)
        plt.imshow(np.squeeze(inp_save))

        plt.subplot(122)
        plt.imshow(np.squeeze(out_save))

        plt.savefig(f'eval_results/eval_{step}_{idx}.png')

    print(f'Evaluation. Step: {step}, Loss: {running_loss / (len(eval_loader))}')


def train(net, optimizer, train_loader, eval_loader, device):
    global_steps = 0
    for epoch in range(args.epochs):
        running_loss = .0
        for idx, data in enumerate(train_loader, 0):
            net.train()

            imgs, _ = data
            imgs = imgs.to(device)

            out, mu, logVar = net(imgs)

            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_steps += 1
            running_loss += loss
            if global_steps % args.log_interval == 0:
                print(f'Epoch: {epoch}, Step: {global_steps}, Loss: {running_loss / (idx + 1)}')

            if global_steps % args.eval_interval == 0:
                eval(net, eval_loader, device, global_steps)
                pass


def init_training():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)

    eval_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = VAE().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    train(net, optimizer, train_loader, eval_loader, device)


if __name__ == '__main__':
    init_training()
