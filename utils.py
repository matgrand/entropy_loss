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

def entropy(x):
    _, counts = th.unique(x, return_counts=True)
    p = counts/len(x)
    # return -th.sum(p*th.log2(p))
    return -th.sum(p*th.log(p))

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

# v1 https://discuss.pyth.org/t/calculating-the-entropy-loss/14510, not working in this case
class HLoss1(nn.Module): # https://discuss.pyth.org/t/calculating-the-entropy-loss/14510
    def __init__(self, ε, max):
        super(HLoss1, self).__init__()
        self.ε, self.max = ε, max
        self.nlevels = int(2*max / ε)//2+1 

    def forward(self, x1, x2=None):
        r = x1 if x2 is None else x1 - x2
        rq = quantize(r, self.ε, self.max)

        # convert to 1-hot encoding
        rq_tmp = th.round(rq/(2*self.ε)).long() + self.nlevels//2
        rq_1hot = F.one_hot(rq_tmp, num_classes=self.nlevels).float()

        b = F.softmax(rq_1hot, dim=-1) * F.log_softmax(rq_1hot, dim=-1)
        b = -1.0 * b.sum() / rq.size(0)
        return b
    
# # v2: https://en.wikipedia.org/wiki/Kernel_density_estimation, kinda working, no quantization
'''# class HLoss2(nn.Module): # https://en.wikipedia.org/wiki/Kernel_density_estimation
#     def __init__(self, ε, max):
#         super(HLoss2, self).__init__()
#         self.ε, self.max = ε, max
#         # self.nlevels = int(2*max / ε)//2+1 
#     def forward(self, x1, x2=None):
#         r = x1 if x2 is None else x1 - x2
#         # rq = quantize(r, self.ε, self.max) # no quantization -> no grad

#         rqs = r
#         ents = 0
#         for rq in rqs:
#             σ = 1*self.ε # width of the gaussian kernel
#             # sample m points from a gaussian
#             m = 300
#             μrq, σrq = th.mean(rq), th.std(rq) # mean and std of the quantized signal
#             samples = th.randn(m) * σrq + μrq # samples from the gaussian
#             likelihoods = normal(samples, μrq, σrq) # likelihoods
#             # Reshape samples to [m, 1] to broadcast against rq
#             samples_reshaped = samples.reshape(-1, 1)
#             # Calculate normal probability for each sample-rq pair
#             probs = normal(samples_reshaped, rq, σ)  # Shape: [m, len(rq)]
#             # Mean across rq dimension
#             p_values = th.mean(probs, dim=1)  # Shape: [m]
#             # Calculate entropy contribution for each sample
#             log_terms = -p_values * th.log(p_values + 1e-8)
#             # Divide by likelihoods and sum
#             ent = th.sum(log_terms / likelihoods)
#             ents += ent/m
#         return ents / len(rq)
'''
class HLoss2(nn.Module): # https://en.wikipedia.org/wiki/Kernel_density_estimation
    def __init__(self, ε, max):
        super(HLoss2, self).__init__()
        self.ε, self.max = ε, max
        # self.nlevels = int(2*max / ε)//2+1 
    def forward(self, x1, x2=None):
        r = x1 if x2 is None else x1 - x2
        
        # Get batch size
        batch_size = r.shape[0]
        σ = 1*self.ε # width of the gaussian kernel
        m = 500
        
        # Compute statistics for each item in batch
        μrqs = th.mean(r, dim=1)  # [batch_size]
        σrqs = th.std(r, dim=1)   # [batch_size]
        
        # Create samples for each item in batch - shape [batch_size, m]
        # We use expand instead of repeat to avoid copying data
        batch_randn = th.randn(batch_size, m, device=r.device)
        samples = batch_randn * σrqs.unsqueeze(1) + μrqs.unsqueeze(1)
        
        # Calculate likelihoods - shape [batch_size, m]
        # Creating a broadcasting-friendly version of μrqs and σrqs
        μrqs_expanded = μrqs.unsqueeze(1)  # [batch_size, 1]
        σrqs_expanded = σrqs.unsqueeze(1)  # [batch_size, 1]
        likelihoods = normal(samples, μrqs_expanded, σrqs_expanded)
        
        # For each batch item and each sample, calculate probability
        # Reshape r to [batch_size, 1, seq_len] and samples to [batch_size, m, 1]
        r_expanded = r.unsqueeze(1)  # [batch_size, 1, seq_len]
        samples_expanded = samples.unsqueeze(2)  # [batch_size, m, 1]
        
        # Calculate probabilities - shape [batch_size, m, seq_len]
        probs = normal(samples_expanded, r_expanded, σ)
        
        # Mean across sequence dimension - shape [batch_size, m]
        p_values = th.mean(probs, dim=2)
        
        # Calculate entropy contribution - shape [batch_size, m]
        log_terms = -p_values * th.log(p_values + 1e-8)
        
        # Divide by likelihoods and sum for each batch item - shape [batch_size]
        ents = th.sum(log_terms / likelihoods, dim=1) / m
        
        # Return average entropy across batch
        return th.mean(ents)

        
    
class HLoss3(nn.Module): # by claude 3.7: prompt: write a pytorch module/loss that calculates the entropy of a quantized signal, make sure the entropy loss is differentiable and can be backprop. The signal is a N timesteps window (shape (batch_size,N,1)), the quantization step is epsilon. Note that the quantization is in the values of the signals, not the time. 
    """
    Calculates the entropy of a quantized signal in a differentiable manner.

    The signal is quantized with a step size of epsilon, and the entropy is calculated
    based on the probability distribution of the quantized values.
    
    To maintain differentiability, soft quantization is used with a temperature parameter
    that controls the smoothness of the quantization.
    
    Args:
        epsilon (float): The quantization step size
        min_val (float, optional): Minimum value for quantization range. Default: -1.0
        max_val (float, optional): Maximum value for quantization range. Default: 1.0
        temperature (float, optional): Temperature for soft quantization. Lower values
            approach hard quantization. Default: 0.1
        eps (float, optional): Small value to avoid log(0). Default: 1e-10
    """
    def __init__(self, ε, max_val=1.0, temperature=0.1, eps=1e-10):
        super().__init__()
        self.ε = ε
        self.max_val = max_val
        self.temperature = temperature
        self.eps = eps
        
        # Calculate the number of quantization levels
        self.num_levels = int(2*max_val / ε)//2 + 1 # int(2*MAX_SIG / EPSI)//2+1
        
        # Create quantization centers
        # self.register_buffer('centers', th.arange(min_val, max_val + ε, ε))
        self.centers = th.arange(-max_val, max_val, 2*ε, requires_grad=False)
    
    def soft_quantize(self, x):
        """
        Performs soft quantization using a differentiable approach.
        
        Args:
            x (torch.Tensor): Input signal of shape (batch_size, N, )
            
        Returns:
            tuple: (soft_quantized_signal, soft_assignment)
                - soft_quantized_signal: The soft quantized signal
                - soft_assignment: The soft assignment to each quantization level
        """
        # Reshape x to (batch_size * N, 1)
        x_shape = x.shape
        x_flat = x.reshape(-1, 1)  # (batch_size * N, 1)
        
        # Calculate distance to each quantization center
        # Result shape: (batch_size * N, num_levels)
        dist = -th.abs(x_flat - self.centers) / self.temperature
        
        # Softmax to get soft assignments to quantization levels
        soft_assignment = F.softmax(dist, dim=1)  # (batch_size * N, num_levels)
        
        # Compute soft quantized values
        # Sum over quantization levels: (batch_size * N, 1)
        soft_quantized = th.matmul(soft_assignment, self.centers.unsqueeze(1))
        
        # Reshape back to original shape
        soft_quantized = soft_quantized.reshape(*x_shape)
        soft_assignment = soft_assignment.reshape(*x_shape, self.num_levels)
        
        return soft_quantized, soft_assignment
    
    def forward(self, x1, x2=None):
        """
        Calculate the differentiable entropy loss.
        
        Args:
            x (th.Tensor): Input signal of shape (batch_size, N, 1)
            
        Returns:
            th.Tensor: The entropy loss (scalar)
        """        

        r = x1 if x2 is None else x1 - x2

        # Get soft assignments to quantization levels
        _, soft_assignment = self.soft_quantize(r)  # (batch_size, timesteps, num_levels)
        
        # Calculate the probability of each quantization level
        # by averaging over the time dimension
        # Shape: (batch_size, num_levels)
        prob = th.mean(soft_assignment, dim=1)
        
        # Calculate entropy for each batch
        # H = -sum(p * log(p))
        entropy = -th.sum(prob * th.log2(prob + self.eps), dim=1)
        
        # Average over batch
        mean_entropy = th.mean(entropy)
        
        return mean_entropy

    
class HLoss4(nn.Module): 
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