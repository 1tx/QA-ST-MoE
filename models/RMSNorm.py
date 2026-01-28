import torch.nn as nn
import torch


class RMSNorm(nn.Module):
    def __init__ (self, hidden_dim,eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


if __name__ == '__main__':
    x = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    x = torch.tensor(x,dtype=torch.float32)
    Rmsnorm = RMSNorm(4)
    x_norm = Rmsnorm(x)
    print(x_norm)
