import os
import torch

def FP16(ema):
    shadow16 = {}
    for k, v in ema.items():
        shadow16[k] = v.detach().cpu().half()
    return shadow16


def saveModel(network, ema, step, optimizer, scaler, lossV, dataset, mode="ema"):
    save_dir = f"./{dataset}checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    if(mode == "all"):
        torch.save(
            {
                "step": step,
                "ema": ema.shadow,
                "model_state": network.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": lossV.item(),
                "scaler": scaler.state_dict(),
            },
            f"{save_dir}/ckpt_step_{step}.pt"
        )
    elif(mode == "ema"):
        torch.save(
            {
                "step": step,
                "ema": FP16(ema.shadow),
                "loss": lossV.item(),
            },
            f"{save_dir}/ckpt_step_{step}_ema.pt"
        )
    elif(mode == "raw"):
        torch.save(
            {
                "step": step,
                "model_state": FP16(network.state_dict()),
                "loss": lossV.item(),
            },
            f"{save_dir}/ckpt_step_{step}_raw.pt"
        )

def loadModel(step, network, ema, dataset, optimizer=None, scaler=None, mode="ema", device="cuda"):
    ckpt = torch.load(f"{dataset}checkpoints/ckpt_step_{step}_{mode}.pt", map_location=device)

    if(mode == "all"):
        network.load_state_dict(ckpt["model_state"])
        ema.shadow = ckpt["ema"]
    elif(mode == "raw"):
        network.load_state_dict(ckpt["model_state"])
    elif(mode == "ema"):
        ema.shadow = ckpt["ema"]
    if(optimizer is not None):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if(scaler is not None):
        scaler.load_state_dict(ckpt["scaler"])

    start_step = ckpt["step"] + 1

    return start_step, optimizer, network

def loadEMA(step, ema, dataset, device="cuda"):
    ckpt = torch.load(f"{dataset}checkpoints/ckpt_step_{step}_ema_fp16.pt", map_location=device)
    ema.shadow = ckpt["ema"]
    return ema