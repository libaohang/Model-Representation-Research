import torch
import math

def dynamic_threshold(x0, p=0.995):
    # x0: (B,C,H,W)
    s = torch.quantile(x0.abs().reshape(x0.size(0), -1), p, dim=1)  # (B,)
    s = torch.clamp(s, min=1.0)  # don't scale up small values
    s = s.view(-1, 1, 1, 1)
    x0 = torch.clamp(x0, -s, s) / s
    return x0

@torch.no_grad()
def ddpmSample(network, T, alphaBars, posterior_log_var, c1, c2, device, shape):
    network.eval()
    x = torch.randn(shape, device=device)

    B = shape[0]

    for t in reversed(range(T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        ab_t = alphaBars[t].view(1,1,1,1)

        eps_hat = network(x, t_batch)

        # x0 estimate from eps prediction
        x0_hat = (x - torch.sqrt(1.0 - ab_t) * eps_hat) / torch.sqrt(ab_t)
        x0_hat = dynamic_threshold(x0_hat, p=0.995)

        mean = c1[t] * x0_hat + c2[t] * x

        if t > 0:
            sigma = torch.exp(0.5 * posterior_log_var[t]).view(1,1,1,1)
            x = mean + sigma * torch.randn_like(x)
        else:
            x = mean

    return x

@torch.no_grad()
def ddimSample(network, T, alphaBars, shape, device, steps=50, eta=0.0):

    B, C, H, W = shape
    x = torch.randn(shape, device=device)

    tSeq = torch.linspace(T-1, 0, steps, device=device).long()

    for i in range(len(tSeq)-1):
        t = tSeq[i]
        tNext = tSeq[i+1]

        tBatch = torch.full((B,), t.item(), device=device, dtype=torch.long)

        ab_t = alphaBars[t].view(1, 1, 1, 1)
        ab_next = alphaBars[tNext].view(1, 1, 1, 1)

        epsHat = network(x, tBatch)

        # x0 estimate
        x0_hat = (x - torch.sqrt(1.0 - ab_t) * epsHat) / torch.sqrt(ab_t)
        x0_hat = x0_hat.clamp(-1.0, 1.0)  

        if eta == 0.0:
            # deterministic DDIM
            x = torch.sqrt(ab_next) * x0_hat + torch.sqrt(1.0 - ab_next) * epsHat
        else:
            # stochastic DDIM 
            # sigma_t = eta * sqrt((1-ab_next)/(1-ab_t)) * sqrt(1 - ab_t/ab_next)
            sigma = eta * torch.sqrt((1 - ab_next) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_next)
            noise = torch.randn_like(x)
            x = torch.sqrt(ab_next) * x0_hat + torch.sqrt(1 - ab_next - sigma**2) * epsHat + sigma * noise

    return x