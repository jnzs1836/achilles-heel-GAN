import argparse
from data.cifar import get_cifar_loader
from models.discriminator import Discriminator
from models.generator import Generator
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
from torch.utils.tensorboard import SummaryWriter
import tqdm as tqdm

def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="/home/jnzs1836/code/CSGY_6763/data/cifar-10")
parser.add_argument("--log_dir", type=str, default="../experiments/cifar-10/logs")
parser.add_argument("--save_dir", type=str, default="../experiments/cifar-10/saves")

parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)

parser.add_argument("--ngpu", type=int, default=0)

parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=2e-4)

parser.add_argument("--device", type=str, default="cpu")

parser.add_argument("--channel", type=int, default=3)
parser.add_argument("--latent_size", type=int, default=128)
parser.add_argument("--generator_feature_size", type=int, default=64)
parser.add_argument("--discriminator_feature_size", type=int, default=64)

args = parser.parse_args()


def build_optimizer(generator, discriminator, lr, beta1):
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    return optimizerG, optimizerD


def train(data_loader, generator, discriminator, latent_size, lr, beta1, n_epochs, device, log_dir, save_dir, args):
    writer = SummaryWriter(log_dir)
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    nz = latent_size
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.


    optimizerG, optimizerD = build_optimizer(generator, discriminator, lr, beta1)

    print("Starting Training Loop...")
    best_epoch = 0
    best_loss_G = -1
    # For each epoch
    saving_delay = 10
    best_generator_state_dict = None
    best_discriminator_state_dict = None
    for epoch in tqdm.tqdm(range(n_epochs)):
        epoch_loss_G = 0
        # For each batch in the dataloader
        for i, data in enumerate(data_loader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_images = data[0].to(device)
            b_size = real_images.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = discriminator(real_images).view(-1)
            # Calculate loss on all-real batch
            loss_D_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            loss_D_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            loss_D_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            loss_D_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            loss_D = loss_D_real + loss_D_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            loss_G = criterion(output, label)
            # Calculate gradients for G
            loss_G.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 1 == 50:
                tqdm.tqdm.write('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, n_epochs, i, len(data_loader),
                         loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))
            # Save Losses for plotting later
            writer.add_scalar("loss/G", loss_G.item(), iters)
            # G_losses.append(loss_G.item())
            writer.add_scalar("loss/D", loss_D.item(), iters)

            # D_losses.append(loss_D.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == n_epochs-1) and (i == len(data_loader)-1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
            iters += 1
            epoch_loss_G += loss_G.item()
        epoch_loss_G = epoch_loss_G / len(data_loader)

        if epoch >= saving_delay:
            ckpt_path = os.path.join(save_dir, "epoch.pkl")
            backup_ckpt_path = os.path.join(save_dir, "backup.pkl")

            if epoch_loss_G < best_loss_G or best_loss_G < 0:
                best_generator_state_dict = generator.state_dict()
                best_discriminator_state_dict = discriminator.state_dict()
                best_loss_G = epoch_loss_G
                best_epoch = epoc
            checkpoint = {
                "args": args.__dict__,
                "epoch": epoch,
                "epoch_generator_state_dict": generator.state_dict(),
                "epoch_discriminator_state_dict": discriminator.state_dict(),
                "best_generator_state_dict": best_generator_state_dict,
                "best_discriminator_state_dict": best_discriminator_state_dict,
                "best_epoch": best_epoch
            }   
            torch.save(checkpoint, ckpt_path)
            torch.save(checkpoint, backup_ckpt_path)



def main(args):
    train_loader, test_loader = get_cifar_loader(args.dataset, args.batch_size)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

    generator = Generator(args.channel, args.latent_size, args.generator_feature_size, args.ngpu)
    discriminator = Discriminator(args.channel, args.discriminator_feature_size, args.ngpu)
    train(train_loader, generator, discriminator, args.latent_size, args.lr, args.beta1, args.n_epochs, device, args.log_dir, args.save_dir,args)

if __name__ == "__main__":
    main(args)