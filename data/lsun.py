import torch
import torchvision
import torchvision.transforms as transforms


def get_lsun_loader(root, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((64, 64)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = batch_size

    classes = ["bedroom_train", "bridge_train", "church_outdoor_train", "classroom_train", "conference_room_train", "dining_room_train", "kitchen_train", "living_room_train", "restaurant_train", "tower_train"]
    classes = ["bedroom_train", "living_room_train"]
    trainset = torchvision.datasets.LSUN(root=root, classes=classes,
                                        transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

    # testset = torchvision.datasets.LSUN(root=root, classes=classes,
    #                                    transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                     shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, trainloader
