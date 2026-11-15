import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from NoiseSchedule import cosineSchedule
from SaveLoad import saveModel

def trainNetwork(network, dataLoader, lossF, optimizer, ema, dataset, T=1000, epochs=100, saveMode="ema", device="cuda"):
    sqrtA_, sqrt1mA_, *_= cosineSchedule(T, device=device)
    network.train()
    step = 0
    lossV = 0
    batchSize = 0

    scaler = GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(epochs + 1):
        for images, _ in dataLoader:

            images = images.to(device)
            batchSize = images.size(0)

            t = torch.randint(0, T, (batchSize,), device=device)

            eps = torch.randn_like(images)

            a = sqrtA_[t].view(batchSize, 1, 1, 1)
            b = sqrt1mA_[t].view(batchSize, 1, 1, 1)

            noisyImages = a * images + b * eps

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                epsHat = network(noisyImages, t)
                lossV = lossF(epsHat, eps)

            scaler.scale(lossV).backward()
            scaler.step(optimizer)
            scaler.update()

            ema.update(network)

            step += 1
            if step % 100 == 0:
                print(f"step {step}: loss {lossV.item():.4f}")

            if step % 10000 == 0:
                saveModel(network, ema, step, optimizer, scaler, lossV, dataset, saveMode)
        
    return network
