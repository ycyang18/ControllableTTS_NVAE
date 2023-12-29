import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from NVAE.utils import Swish, ChannelSELayer1d, reparameterize, kl, kl_2

class EncoderResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim), 
            Swish(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        )
            #ChannelSELayer1d(dim))
    def forward(self, x):
        return x + 0.1 * self._seq(x)

class DecoderResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Use the same residual block as encoder, bc no channel expansion for now
        self._seq = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim), 
            Swish(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        return x + 0.1 * self._seq(x)

class DownsamplingConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=5, padding=0),
            nn.Conv1d(out_channel, out_channel, kernel_size=5, padding=0),
            nn.BatchNorm1d(out_channel),
            Swish(),
            nn.Conv1d(out_channel, 1, kernel_size=3, padding=1),
            nn.BatchNorm1d(1),
            Swish()
        )
    def forward(self, x):
        return self._seq(x)

class DownsamplingConvBlock_mod(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=5, padding=0),
            #nn.BatchNorm1d(out_channel),
            Swish(),
            #nn.Dropout(p=0.5),
            nn.Conv1d(out_channel, out_channel, kernel_size=5, padding=0),
            #nn.BatchNorm1d(out_channel),
            Swish(),
            #nn.Dropout(p=0.5),
            nn.Conv1d(out_channel, 1, kernel_size=3, padding=1),
            #nn.BatchNorm1d(out_channel),
            Swish(),
            #nn.Dropout(p=0.5),
            #nn.Conv1d(out_channel, 1, kernel_size=3, padding=1),
            #nn.BatchNorm1d(1),
            #Swish(),
            #nn.Dropout(p=0.5),
            #nn.Conv1d(out_channel, 1, kernel_size=1, padding=0),
            #nn.BatchNorm1d(out_channel),
            #Swish(),
            #nn.Dropout(p=0.5),
            #nn.Conv1d(out_channel, 1, kernel_size=3, padding=1),
            #nn.BatchNorm1d(1),
            #Swish(),
            #nn.Dropout(p=0.5),
            #nn.Conv1d(out_channel, 1, kernel_size=3, padding=1),
            #nn.BatchNorm1d(out_channel),
            #Swish(),
            #nn.Dropout(p=0.5),
            #nn.Conv1d(out_channel, 1, kernel_size=1, padding=0),
            #nn.BatchNorm1d(1),
            #Swish(),
            #nn.Dropout(p=0.5),
        )
    def forward(self, x):
        return self._seq(x)

class UpsamplingConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, padding=1),
            nn.Conv1d(out_channel, out_channel, kernel_size=1, stride=1, padding=1),
            nn.Conv1d(out_channel, out_channel, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm1d(out_channel),
            Swish(),
            nn.Conv1d(out_channel, 1, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm1d(1),
            Swish()
        )
    def forward(self, x):
        return self._seq(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim), 
            Swish(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1)
            #SELayer(dim)
        )
    def forward(self, x):
        return x + 0.1 * self._seq(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self._seq = nn.Sequential(
            # control kernal_size & padding: downsampling the sequence, now:seq=64(k=3,p=1)
            nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.Conv1d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channel), 
            Swish(),
            nn.Conv1d(out_channel, 1, kernel_size=3, padding=1),
            nn.BatchNorm1d(1), 
            Swish()
        )
    def forward(self, x):
        return self._seq(x)


#----------------------------------------------------------------------------------------------#
# 4-level-NVAE blocks

class DownsamplingConvBlock4L(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=5, padding=0),
            nn.Conv1d(out_channel, out_channel, kernel_size=5, padding=0),
            nn.Conv1d(out_channel, out_channel, kernel_size=5, padding=0),
            nn.BatchNorm1d(out_channel),
            Swish(),
            nn.Conv1d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channel),
            Swish()
        )
    def forward(self, x):
        return self._seq(x)

class UpsamplingConvBlock4L(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, padding=1),
            nn.Conv1d(out_channel, out_channel, kernel_size=1, stride=1, padding=1),
            nn.Conv1d(out_channel, 1, kernel_size=1, stride=1, padding=1),
            nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm1d(1),
            Swish(),
            nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=1),
            nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm1d(1),
            Swish()
        )
    def forward(self, x):
        return self._seq(x)

class DecoderResidualBlock4L(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Use the same residual block as encoder, bc no channel expansion for now
        self._seq = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim), 
            Swish(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return x + 0.1 * self._seq(x)

class EncoderResidualBlock4L(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim), 
            Swish(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        )
            #ChannelSELayer1d(dim))
    def forward(self, x):
        return x + 0.1 * self._seq(x)


class ResidualBlock4L(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim), 
            Swish(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1)
            #SELayer(dim)
        )
    def forward(self, x):
        return x + 0.1 * self._seq(x)




#----------------------------------------------------------------------------------------------#
# 4-level-NVAE blocks

