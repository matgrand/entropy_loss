import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableQuantizedEntropyLoss(nn.Module):
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
    def __init__(self, epsilon, min_val=-1.0, max_val=1.0, temperature=0.1, eps=1e-10):
        super().__init__()
        self.epsilon = epsilon
        self.min_val = min_val
        self.max_val = max_val
        self.temperature = temperature
        self.eps = eps
        
        # Calculate the number of quantization levels
        self.num_levels = int((max_val - min_val) / epsilon) + 1
        
        # Create quantization centers
        self.centers = torch.arange(min_val, max_val + epsilon, epsilon, requires_grad=False)
    
    def soft_quantize(self, x):
        """
        Performs soft quantization using a differentiable approach.
        
        Args:
            x (torch.Tensor): Input signal of shape (batch_size, N, 1)
            
        Returns:
            tuple: (soft_quantized_signal, soft_assignment)
                - soft_quantized_signal: The soft quantized signal
                - soft_assignment: The soft assignment to each quantization level
        """
        # Reshape x to (batch_size * N, 1)
        batch_size, timesteps, _ = x.shape
        x_flat = x.reshape(-1, 1)  # (batch_size * N, 1)
        
        # Calculate distance to each quantization center
        # Result shape: (batch_size * N, num_levels)
        dist = -torch.abs(x_flat - self.centers) / self.temperature
        
        # Softmax to get soft assignments to quantization levels
        soft_assignment = F.softmax(dist, dim=1)  # (batch_size * N, num_levels)
        
        # Compute soft quantized values
        # Sum over quantization levels: (batch_size * N, 1)
        soft_quantized = torch.matmul(soft_assignment, self.centers.unsqueeze(1))
        
        # Reshape back to original shape
        soft_quantized = soft_quantized.reshape(batch_size, timesteps, 1)
        soft_assignment = soft_assignment.reshape(batch_size, timesteps, -1)
        
        return soft_quantized, soft_assignment
    
    def forward(self, x):
        """
        Calculate the differentiable entropy loss.
        
        Args:
            x (torch.Tensor): Input signal of shape (batch_size, N, 1)
            
        Returns:
            torch.Tensor: The entropy loss (scalar)
        """
        batch_size, timesteps, _ = x.shape
        
        # Get soft assignments to quantization levels
        _, soft_assignment = self.soft_quantize(x)  # (batch_size, timesteps, num_levels)
        
        # Calculate the probability of each quantization level
        # by averaging over the time dimension
        # Shape: (batch_size, num_levels)
        prob = torch.mean(soft_assignment, dim=1)
        
        # Calculate entropy for each batch
        # H = -sum(p * log(p))
        entropy = -torch.sum(prob * torch.log2(prob + self.eps), dim=1)
        
        # Average over batch
        mean_entropy = torch.mean(entropy)
        
        return mean_entropy

# Example usage
if __name__ == "__main__":
    # Create sample data: batch of 8, 100 timesteps, 1 channel
    batch_size, timesteps = 8, 100
    # Make sure the input requires gradients
    # x = torch.randn(batch_size, timesteps, 1, requires_grad=True)
    x = torch.randn(batch_size, timesteps, requires_grad=True)
    
    # Define epsilon (quantization step)
    epsilon = 0.05
    
    # Create the entropy loss module
    entropy_loss = DifferentiableQuantizedEntropyLoss(epsilon=epsilon)
    
    # Calculate entropy
    loss = entropy_loss(x)
    print(f"Entropy: {loss.item()}")
    
    # Demonstrate backpropagation
    loss.backward()
    print("Backpropagation successful!")
    
    # We can also minimize or maximize entropy during training:
    # To minimize entropy (make distribution more peaked):
    optimizer = torch.optim.Adam([x], lr=0.01)
    optimizer.zero_grad()
    loss = entropy_loss(x)
    loss.backward()
    optimizer.step()
    
    # # To maximize entropy (make distribution more uniform):
    # optimizer.zero_grad()
    # (-loss).backward()
    # optimizer.step()