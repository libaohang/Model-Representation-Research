import torch
import random
from SaveLoad import loadModel
from Sampling import ddimSample, ddpmSample
from NoiseSchedule import cosineSchedule
from torchvision.utils import save_image
from UNet import UNet
from EMA import EMA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

channels = [64, 128, 256, 256]
attention = [False, False, True, True]

network = UNet(3, channels, attention, 2, 2, 64)
network.to(device)

ema = EMA(network, decay=0.99995)

step = 350500

loadModel(step, network, ema, "CIFAR10-", mode = "ema", device=device)

T = 1000

def setSeed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def ddim(applyEMA=False, seed=None):
    if seed is not None:
        setSeed(seed)

    if(applyEMA):
        ema.applyShadow(network)
    *_, alphaBars = cosineSchedule(T, device=device)
    samples = ddimSample(
        network=network,
        T=T,
        alphaBars=alphaBars,
        shape=(64, 3, 32, 32),
        device=device,
        steps=100,
        eta=0.0
    )
    if(applyEMA):
        ema.restore(network)

    # save
    samples_vis = (samples + 1) / 2
    samples_vis = samples_vis.clamp(0, 1)
    save_image(samples_vis, f"samples_{int(step/1000)}k_ddim100" + ("_ema" if (applyEMA) else "") + (f"_seed{seed}" if seed is not None else "") + ".png", nrow=8)
    print(f"Saved samples_{int(step/1000)}k_ddim100" + ("_ema" if (applyEMA) else "") + (f"_seed{seed}" if seed is not None else "") + ".png")

def ddpm(applyEMA=False, seed=None):
    if seed is not None:
        setSeed(seed)

    if(applyEMA):
        ema.applyShadow(network)
    alphaBars, posterior_log_var, c1, c2 = cosineSchedule(T, device=device, full=True)
    samples = ddpmSample(
        network=network,
        T=T,
        alphaBars=alphaBars,
        posterior_log_var=posterior_log_var, 
        c1=c1, 
        c2=c2,
        device=device,
        shape=(10, 3, 32, 32)
    )
    if(applyEMA):
        ema.restore(network)

    samples_vis = (samples + 1) / 2
    samples_vis = samples_vis.clamp(0, 1)
    save_image(samples_vis, 
        f"samples_{int(step/1000)}k_ddpm" + ("_ema" if (applyEMA) else "") + (f"_seed{seed}" if seed is not None else "") + ".png",
        nrow=10, 
        padding=2, 
        pad_value=1.0
    )
    print(f"Saved samples_{int(step/1000)}k_ddpm" + ("_ema" if (applyEMA) else "") + (f"_seed{seed}" if seed is not None else "") + ".png")

if __name__ == "__main__":
    ddpm(applyEMA=True, seed = 30)