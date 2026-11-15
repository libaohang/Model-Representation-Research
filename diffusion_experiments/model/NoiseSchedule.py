import torch
import math

def precompute(betas, alphas, alphaBars, eps=1e-20):
    sqrtA_ = torch.sqrt(alphaBars)
    sqrt1mA_ = torch.sqrt((1.0 - alphaBars).clamp(min=0.0))
    sqrtAInv = torch.sqrt(1.0 / alphas.clamp(min=eps))
    epsCoeff = betas / sqrt1mA_.clamp(min=eps)

   # posterior variance: beta_t * (1 - alphaBar_{t-1}) / (1 - alphaBar_t)
    alphaBars_prev = torch.cat([alphaBars.new_tensor([1.0]), alphaBars[:-1]])
    posterior_variance = betas * (1.0 - alphaBars_prev) / (1.0 - alphaBars).clamp(min=eps)
    posterior_variance = posterior_variance.clamp(min=1e-20)

    sqrtPosteriorVar = torch.sqrt(posterior_variance)
    return sqrtA_, sqrt1mA_, sqrtAInv, epsCoeff, sqrtPosteriorVar, alphaBars

def precomputeFull(betas, alphas, alphaBars, eps=1e-20):
    # alphaBars: length T, alphaBars[0] should be 1.0
    alphaBars_prev = torch.cat([alphaBars.new_tensor([1.0]), alphaBars[:-1]])

    denom = (1.0 - alphaBars)  

    posterior_var = betas * (1.0 - alphaBars_prev) / denom.clamp(min=eps)
    posterior_var = posterior_var.clamp(min=1e-20)

    posterior_log_var = torch.log(posterior_var)

    c1 = (torch.sqrt(alphaBars_prev) * betas) / denom.clamp(min=eps)
    c2 = (torch.sqrt(alphas) * (1.0 - alphaBars_prev)) / denom.clamp(min=eps)

    posterior_var[0] = 0.0
    posterior_log_var[0] = -float("inf")  
    c1[0] = 1.0
    c2[0] = 0.0

    posterior_log_var = posterior_log_var.clamp(min=-20.0, max=0.0)

    return alphaBars, posterior_log_var, c1, c2

def linearSchedule(T=1000, beta1 = 1e-4, betaT = 2e-2, device="cuda"):
    betas = torch.linspace(beta1, betaT, T, device=device)
    alphas = 1.0 - betas
    alphaBars = torch.cumprod(alphas, dim=0)
    return precompute(betas, alphas, alphaBars)

def cosineSchedule(T=1000, s=0.008, device="cuda", full=False):
    t = torch.linspace(0, 1, T + 1, device=device, dtype=torch.float64)
    alphaBars = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
    alphaBars = (alphaBars / alphaBars[0])[:-1]

    alphas = torch.ones(T, device=device, dtype=torch.float64)
    alphas[1:] = alphaBars[1:] / alphaBars[:-1]  # aBar[t] / aBar[t-1]

    betas = 1.0 - alphas
    betas = betas.clamp(1e-8, 0.999)

    alphaBars = alphaBars.to(torch.float32)
    betas = betas.to(torch.float32)
    alphas = alphas.to(torch.float32)

    if full:
        return precomputeFull(betas, alphas, alphaBars)
    else:
        return precompute(betas, alphas, alphaBars)

