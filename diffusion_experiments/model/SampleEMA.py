import torch
from SaveLoad import loadEMA
from Sampling import ddimSample, ddpmSample
from NoiseSchedule import cosineSchedule
from torchvision.utils import save_image
from UNet import UNet
from EMA import EMA

def ddim(ema, network, dataset, shape, modelSteps, steps, rows, device="cuda"):
    
    ema.applyShadow(network)
    *_, alphaBars = cosineSchedule(1000, device=device)
    samples = ddimSample(
        network=network,
        T=1000,
        alphaBars=alphaBars,
        shape=shape,
        device=device,
        steps=steps,
        eta=0.0
    )
    ema.restore(network)

    samples_vis = (samples + 1) / 2
    samples_vis = samples_vis.clamp(0, 1)
    save_image(samples_vis, f"{dataset}_samples_{int(modelSteps/1000)}k_ddim100_ema.png", nrow=rows)
    print(f"Saved {dataset}_samples_{int(modelSteps/1000)}k_ddim100_ema.png")

def ddpm(ema, network, dataset, shape, modelSteps, rows, device="cuda"):
    ema.applyShadow(network)
    alphaBars, posterior_log_var, c1, c2 = cosineSchedule(1000, device=device, full=True)
    samples = ddpmSample(
        network=network,
        T=1000,
        alphaBars=alphaBars,
        posterior_log_var=posterior_log_var, 
        c1=c1, 
        c2=c2,
        device=device,
        shape=shape
    )
    ema.restore(network)

    samples_vis = (samples + 1) / 2
    samples_vis = samples_vis.clamp(0, 1)
    save_image(samples_vis, f"{dataset}_samples_{int(modelSteps/1000)}k_ddpm_ema.png", nrow=rows)
    print(f"Saved {dataset}_samples_{int(modelSteps/1000)}k_ddpm_ema.png")

def sampleCIFAR10(type, rows):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channels = [64, 128, 256, 256]
    attention = [False, False, True, True]

    network = UNet(3, channels, attention, 2, 2, 64)
    network.to(device)

    ema = EMA(network, decay=0.99995)

    shape = (rows ** 2, 3, 32, 32)

    step = 350500

    ema = loadEMA(step, ema, "CIFAR10", device=device)

    if(type == "ddim"):
        ddim(ema, network, "CIFAR10", shape, step, 100, rows, device=device)
    else:
        ddpm(ema, network, "CIFAR10", shape, step, rows, device=device)

def sampleCelebA(type, rows):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channels = [64, 128, 256, 512, 512]
    attention = [False, False, True, True, False]

    network = UNet(3, channels, attention, 3, 2, 64)
    network.to(device)

    ema = EMA(network, decay=0.99995)

    shape = (rows ** 2, 3, 64, 64)

    step = 229407

    ema = loadEMA(step, ema, "CelebA", device=device)

    if(type == "ddim"):
        ddim(ema, network, "CelebA", shape, step, 100, rows, device=device)
    else:
        ddpm(ema, network, "CelebA", shape, step, rows, device=device)

if __name__ == "__main__":
    sampleCIFAR10("ddim", 8)
    sampleCelebA("ddim", 4)