import torch as th
# import numpy as np
from math import pi as π
import math
from numpy.random import uniform as uni
from torch import nn
import torch.nn.functional as F

def normal(x, μ, σ): return th.exp(-0.5*((x-μ)/σ)**2)/(σ*math.sqrt(2*π))

# def quantize(x, ε): return 2*ε*np.round(x/(2*ε)) # quantize to multiples of ε
def quantize(x, ε, max): 
    x = th.clamp(x, -max, max)
    xq= 2*ε*th.round(x/(2*ε)) # quantize to multiples of ε
    with th.no_grad(): # straight through estimator, ignore quantization in the backward pass
        delta = x-xq
    return x-delta

# def entropy(x):
#     _, counts = np.unique(x, return_counts=True)
#     p = counts/len(x)
#     # return -np.sum(p*np.log2(p))
#     return -np.sum(p*np.log(p))

def entropy(x):
    _, counts = th.unique(x, return_counts=True)
    p = counts/len(x)
    # return -th.sum(p*th.log2(p))
    return -th.sum(p*th.log(p))

# def create_random_signal(n_samples, n_frqs=2):
#     # generate random frequencies
#     fs = uni(0, 5, n_frqs)
#     As = uni(0.8, 1, n_frqs)
#     φs = uni(0, 2*π, n_frqs)
#     t = np.linspace(0, 1, n_samples)
#     # generate the signal
#     x = np.sum([As[i]*np.sin(2*π*fs[i]*t+φs[i]) for i in range(n_frqs)], axis=0).astype(np.float32)
#     return x

def create_random_signal(n_samples, n_frqs=2):
    # generate random frequencies
    fs = th.rand(n_frqs)*5
    As = 0.8 + 0.2*th.rand(n_frqs)
    φs = 2*π*th.rand(n_frqs)
    t = th.linspace(0, 1, n_samples)
    # generate the signal
    x = th.sum(th.stack([As[i]*th.sin(2*π*fs[i]*t+φs[i]) for i in range(n_frqs)]), axis=0)
    return x


####################################################################################################
# torch modules for entropy loss

class HLoss1(nn.Module): # https://discuss.pytorch.org/t/calculating-the-entropy-loss/14510
    def __init__(self, ε, max):
        super(HLoss1, self).__init__()
        self.ε, self.max = ε, max
        self.nlevels = int(2*max / ε)//2+1 
    def forward(self, x1, x2=None):
        r = x1 - x2 if x2 is not None else x1
        rq = quantize(r, self.ε, self.max)

        # convert to 1-hot encoding
        rq_tmp = th.round(rq/(2*self.ε)).long() + self.nlevels//2
        rq_1hot = F.one_hot(rq_tmp, num_classes=self.nlevels).float()

        b = F.softmax(rq_1hot, dim=-1) * F.log_softmax(rq_1hot, dim=-1)
        b = -1.0 * b.sum() / rq.size(0)
        return b
    
class HLoss2(nn.Module): # https://en.wikipedia.org/wiki/Kernel_density_estimation
    def __init__(self, ε, max):
        super(HLoss2, self).__init__()
        self.ε, self.max = ε, max
        # self.nlevels = int(2*max / ε)//2+1 
    def forward(self, x1, x2=None):
        r = x1 - x2 if x2 is not None else x1
        # rq = quantize(r, self.ε, self.max) # no quantization -> no grad
        rq = r

        # rq = rq-th.mean(rq)

        σ = 1*self.ε # width of the gaussian kernel
        # sample m points from a isotropic gaussian
        m = 300
        
        μ1, σ1 = th.mean(rq), th.std(rq)
        samples = th.randn(m) * σ1 + μ1
        likelihoods = normal(samples, μ1, σ1)
        #calculate pdf of the quantized signal
        ent = 0 # TODO: vectorize this shit
        for s,l in zip(samples, likelihoods):
            p = th.mean(normal(s, rq, σ))
            ent += -p*th.log(p+1e-8) / l
        return ent/m
    
class HLoss3(nn.Module): 
    def dentropy(rq, b=10.0, ε=0.1):
        symbols, counts = np.unique(rq, return_counts=True)
        p = counts/len(rq)
        # logp = np.log2(p + 1e-8)
        logp = np.log(p + 1e-8)
        H = -np.sum(p*logp) # entropy
        sizer = len(rq)
        DH = 0
        for j in range(len(symbols)):
            DH += (1+logp[j])*b / (sizer*ε**b) * (rq-symbols[j])**(b-1) / (((rq-symbols[j])/ε)**b+1)**2
        return H, DH