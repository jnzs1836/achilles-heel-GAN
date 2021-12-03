import torch
import torch.nn as nn
import argparse
from models.generator import Generator
from models.discriminator import Discriminator
from torchvision import utils 
import os


parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint_path", type=str, default="../experiments/saves/epoch.pkl")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--target_dir", type=str, default="../experiments/generated")
parser.add_argument("--num_images", type=int, default=1000)

args = parser.parse_args()

def save_image(image, target_path):
    utils.save_image(image, target_path)

def generate(generator, batch_size, latent_size, num_images, target_dir):
    num_iter = num_images // batch_size
    idx = 0
    padding_num = len(str(num_images)) + 2
    for i in range(num_iter):
        noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
        fake = generator(noise)
        for j in range(fake.size(0)):
            idx += 1
            target_path = os.path.join(target_dir, "image-{}.png".format(str(idx).zfill(padding_num)))
            save_image(fake[j], target_path)


def main(args):
    checkpoint = torch.load(args.checkpoint_path)
    model_args = checkpoint['args']

    device = torch.device("cuda:0" if (torch.cuda.is_available() and model_args['ngpu'] > 0) else "cpu")

    
    channel = model_args['channel']
    latent_size = model_args['latent_size']
    generator_feature_size = model_args['generator_feature_size']
    discriminator_feature_size = model_args['discriminator_feature_size']
    ngpu = model_args['ngpu']
    batch_size = model_args['batch_size']
    num_images = args.num_images
    target_dir = args.target_dir

    generator = Generator(channel, latent_size, generator_feature_size, ngpu)
    discriminator = Discriminator(channel, discriminator_feature_size, ngpu)
    generator.load_state_dict(checkpoint['bet'])
    discriminator.load_state_dict()
    generate(generator, batch_size, latent_size, num_images, target_dir)
    

if __name__ == "__main__":
    main(args)
