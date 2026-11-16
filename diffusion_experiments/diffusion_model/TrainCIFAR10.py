import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from UNet import UNet
from EMA import EMA
from TrainNetwork import trainNetwork

def trainCIFAR10():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    torch.backends.cudnn.benchmark = True

    channels = [64, 128, 256, 256]
    attention = [False, False, True, True]

    network = UNet(3, channels, attention, 2, 2, 64)
    network = network.to(device)

    ema = EMA(network, decay=0.99995)

    transform = transforms.Compose([
        transforms.ToTensor(),                     # [0,1]
        transforms.Lambda(lambda x: x * 2.0 - 1.0) # [-1,1]
    ])

    cifar10 = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    loader = DataLoader(
        cifar10,
        batch_size=100,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    lossF = F.mse_loss

    optimizer = torch.optim.AdamW(network.parameters(), lr=2e-4, weight_decay=0.0)

    network = trainNetwork(network, loader, lossF, optimizer, ema, "CIFAR10", 1000, 700, device)

if __name__ == "__main__":
    trainCIFAR10()

