import torch
from torch import nn

class EpsilonNetFromScore(nn.Module):

    def __init__(self, ou_dist, alphas_cumprod, bias=False, bias_mean=0, bias_std=1e-2):
        super().__init__()
        self.ou_dist = ou_dist
        self.alphas_cumprod = alphas_cumprod
        self.bias = bias
        self.bias_mean = bias_mean
        self.bias_std = bias_std

    def forward(self, x, t):
        z = x.clone().detach()
        z.grad = None
        z.requires_grad = True
        loss = self.ou_dist(self.alphas_cumprod[int(t)]).log_prob(z)
        loss.sum().backward()
        score = z.grad
        z = z.detach()
        coeff = -(1 - self.alphas_cumprod[int(t)])**.5
        if not self.bias:
            return coeff * score
        return coeff * (score + torch.randn_like(score)*self.bias_std + self.bias_mean)


class Expandednet(nn.Module):
    # for DDRM
    def __init__(self,  base_net, expanded_size):
        super().__init__()
        self.base_net = base_net
        self.expanded_size = expanded_size

    def forward(self, x, t):
        return self.base_net(x.flatten(1, len(x.shape)-1), t[0]).reshape(x.shape[0], *self.expanded_size)
