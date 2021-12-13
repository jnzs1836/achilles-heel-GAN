import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import tqdm

import argparse

def get_data_loader(root, batch_size, workers=2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
    return dataloader


class FeatureEncoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model = models.googlenet(pretrained=True)
    
    def forward(self, x):
        return self.model(x)

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--target_dir", type=str, default="../experiments/lsun/generated")
parser.add_argument("--save_path", type=str, default="../experiments/lsun/features/dcgan.pth")
parser.add_argument("--num_images", type=int, default=1000)

args = parser.parse_args()

def main(args):
    feature_extractor = FeatureEncoder().cuda()
    dataloader = get_data_loader(args.target_dir, args.batch_size)
    all_features = []
    for batch in tqdm.tqdm(dataloader):
        batch = batch[0]
        batch = batch.cuda()
        features = feature_extractor(batch)
        all_features.append(features)
    all_features = torch.cat(all_features, dim=0)
    checkpoint = {
        "features": all_features
    }
    torch.save(checkpoint, args.save_path)

if __name__ == "__main__":
    main(args)