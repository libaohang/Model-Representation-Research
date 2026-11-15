import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from UNet import UNet
from EMA import EMA
from TrainNetwork import trainNetwork

def trainCelebA():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    torch.backends.cudnn.benchmark = True

    channels = [64, 128, 256, 512, 512]
    attention = [False, False, True, True, False]

    network = UNet(3, channels, attention, 3, 2, 64)
    network = network.to(device)

    ema = EMA(network, decay=0.99995)

    transform = transforms.Compose([
        transforms.CenterCrop(178),        
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    celebA = datasets.CelebA(root="./data", split="train", download=True, transform=transform)

    loader = DataLoader(
        celebA,
        batch_size=100,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    lossF = F.mse_loss

    optimizer = torch.optim.AdamW(network.parameters(), lr=2e-4, weight_decay=0.0)

    network = trainNetwork(network, loader, lossF, optimizer, ema, "CelebA", 1000, 180, "raw", device)

if __name__ == "__main__":
    trainCelebA()

