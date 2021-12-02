import torch
import torchvision
import torchvision.transforms as transforms


def get_cifar_loader(root, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((64, 64)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = batch_size

    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader