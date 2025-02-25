import torch as th
import numpy as np
from math import pi as π

def normal(x, μ, σ): return torch.exp(-0.5*((x-μ)/σ)**2)/(σ*np.sqrt(2*π))

def quantize_np(x, ε): return 2*ε*np.round(x/(2*ε)) # quantize to multiples of ε
def quantize_th(x, ε): return 2*ε*th.round(x/(2*ε)) # quantize to multiples of ε

def entropy_np(x):
    _, counts = np.unique(x, return_counts=True)
    p = counts/len(x)
    # return -np.sum(p*np.log2(p))
    return -np.sum(p*np.log(p))

def entropy_th(x):
    _, counts = th.unique(x, return_counts=True)
    p = counts/len(x)
    # return -th.sum(p*th.log2(p))
    return -th.sum(p*th.log(p))

