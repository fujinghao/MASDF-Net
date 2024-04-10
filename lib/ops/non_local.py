import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, channel, ratio):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channel, channel // ratio, 1)
        self.key = nn.Conv2d(channel, channel // ratio, 1)
        self.value = nn.Conv2d(channel, channel, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        b, c, h, w = X.size()

        Q = self.query(X) #[b,c/r, h ,w]
        q = Q.reshape(b, c, h * w).permute(0, 2, 1) #[b, h*w, c/r]

        K = self.key(X) #[b,c/r, h ,w]
        k = K.reshape(b, c, h * w).permute(0, 2, 1) #[b, h*w, c/r]

        a = self.softmax(torch.bmm(q, k.permute(0, 2, 1)))  #[b, h*w, h*w]

        V = self.value(X)
        v = V.reshape(b, c, h * w).permute(0, 2, 1) #[b, h*w, c]
        x = torch.bmm(a, v).permute(0, 2, 1).reshape(b, c, h, w) #[b,c , h ,w]
        return X + x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, channel, ratio):
        super(MultiHeadCrossAttention, self).__init__()
        self.query = nn.Conv2d(channel, channel // ratio, 1)
        self.key = nn.Conv2d(channel, channel // ratio, 1)
        self.value = nn.Conv2d(channel, channel, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X, Y):
        b, c, h, w = Y.size()
        Q = self.query(X) #[b,c/r, h ,w]
        b1, c1, h1, w1 = Q.size()
        q = Q.reshape(b1, c1, h1 * w1).permute(0, 2, 1) #[b, h*w, c/r]

        K = self.key(X) #[b,c/r, h ,w]
        k = K.reshape(b1, c1, h1 * w1).permute(0, 2, 1) #[b, h*w, c/r]

        a = self.softmax(torch.bmm(q, k.permute(0, 2, 1)))  #[b, h*w, h*w]

        V = self.value(Y)
        v = V.reshape(b, c, h * w).permute(0, 2, 1) #[b, h*w, c]
        x = torch.bmm(a, v).permute(0, 2, 1).reshape(b, c, h, w) #[b,c , h ,w]
        return Y + x

if __name__ == '__main__':
    model = MultiHeadCrossAttention(128,8).cuda()
    from thop import profile
    import torch
    input1 = torch.randn(1, 128, 124, 124).to('cuda')
    input2 = torch.randn(1, 128, 124, 124).to('cuda')

    macs, params = profile(model, inputs=(input1,input2))
    print('macs:', macs / 1000000000)
    print('params:', params / 1000000)