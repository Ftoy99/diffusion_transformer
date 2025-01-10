import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps  # epsilon to avoid zeroes , small value to not affect calc
        self.scale = nn.Parameter(torch.ones(d_model)) # γ
        self.bias = nn.Parameter(torch.zeros(d_model)) # β

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # mean,avg
        variance = x.var(dim=-1, unbiased=False, keepdim=True)  # avg also| var =  sum [(numb-mean)/count(numb)]^2
        normalized_x = (x - mean) / torch.sqrt(variance + self.eps)  # subtract mean and div by root of variance
        return normalized_x * self.scale + self.bias  # if we 2 aggressive scale them up or down and shift to translate


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.dim = dim
        self.eps = eps # epsilon to avoid zerroes
        self.scale = torch.nn.Parameter(torch.ones(dim)) # γ
        self.bias = nn.Parameter(torch.zeros(dim)) # β
    def forward(self, x):
        # Calculate the RMS of the input tensorx
        xsqrd = x ** 2
        mean = torch.mean(xsqrd, dim=self.dim, keepdim=True)
        rms = torch.sqrt(mean + self.eps)
        normalized_x = x / rms # div by mean root 2
        output = normalized_x * self.scale + self.bias # scale in case we are aggressive
        return output