from data.lsun import get_lsun_loader
from data.cifar import get_cifar_loader
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import tqdm

import argparse

class FeatureEncoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model = models.googlenet(pretrained=True)
    
    def forward(self, x):
        return self.model(x)

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--target_dir", type=str, default="../experiments/lsun/generated")
parser.add_argument("--save_path", type=str, default="../experiments/lsun/features/bedroom_living.pth")
parser.add_argument("--num_images", type=int, default=1000)

args = parser.parse_args()

def main(args):
    feature_extractor = FeatureEncoder().cuda()
    dataloader, _ = get_cifar_loader(args.target_dir, args.batch_size)
    all_features = []
    all_labels = []
    for batch in tqdm.tqdm(dataloader):
        all_labels.append(labels)
        labels = batch[1]
        images = batch[0]
        images = images.cuda()
        features = feature_extractor(images)
        all_features.append(features.detach())
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    checkpoint = {
        "features": all_features,
        "labels": all_labels
    }
    torch.save(checkpoint, args.save_path)

if __name__ == "__main__":
    main(args)
