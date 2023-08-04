from math import log

import torch
from torch import nn
import torch.nn.functional as F

from ...localization import get_2d_mask


class SparseMaxPool(nn.Module):
    def __init__(self, group_size, num_clips):
        super(SparseMaxPool, self).__init__()
        num_group = int(log(num_clips / (2 * group_size), 2)) + 1
        pooling_counts = [2 * group_size - 1] + [group_size] * (num_group - 1)
        
        mask2d, maskij = get_2d_mask(num_clips, group_size, num_group)
        mask2d = torch.tensor(mask2d)
        maskij = maskij[1:] # the main diagnal needs no pooling

        poolers = [nn.MaxPool1d(2,1) for _ in range(pooling_counts[0])]
        for c in pooling_counts[1:]:
            poolers.extend(
                [nn.MaxPool1d(3,2)] + [nn.MaxPool1d(2,1) for _ in range(c - 1)]
            )

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij
        self.poolers = poolers

    def forward(self, x):
        B, D, num_clips = x.shape
        map2d = x.new_zeros(B, D, num_clips, num_clips)
        map2d[:, :, range(num_clips), range(num_clips)] = x
        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x)
            map2d[:, :, i, j] = x
        return map2d


class SMINSparseMeanPool(nn.Module):
    def __init__(self, group_size, num_clips, C=4):
        super(SMINSparseMeanPool, self).__init__()
        self.C = C
        num_group = int(log(num_clips / (2 * group_size), 2)) + 1
        pooling_counts = [2 * group_size - 1] + [group_size] * (num_group - 1)
        
        mask2d, maskij = get_2d_mask(num_clips, group_size, num_group)
        mask2d = torch.tensor(mask2d)
        maskij = maskij[1:] # the main diagnal needs no pooling

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij
    
    def forward(self, x):
        B, D, num_clips = x.shape
        map2d = x.new_zeros(B, D, self.C, num_clips, num_clips)
        for s in range(num_clips):
            for e in range(s, num_clips):
                if (e + 1 - s < self.C):
                    for c in range(self.C):
                        map2d[:, :, c, s, e] = x[:, :, s: e + 1].mean(2)
                else:
                    indices = torch.linspace(s, e + 1, self.C + 1).long()
                    for c in range(self.C):
                        ss = indices[c]
                        ee = indices[c+1]
                        map2d[:, :, c, s, e] = x[:, :, ss: ee].mean(2)
        map2d = map2d * self.mask2d
        return map2d

    # Failed optimization
    # def forward(self, x):
    #     B, D, num_clips = x.shape
    #     map2d = x.new_zeros(B, D, self.C, num_clips, num_clips)
    #     for c in range(self.C):
    #         map2d[:, :, c, range(num_clips), range(num_clips)] = x
    #     for i, j in self.maskij:
    #         kernel_size = j.start - i.start
    #         stride = i.step
    #         num_clips_out = len(i)
            
    #         y_group = F.avg_pool1d(x, max(1, kernel_size // self.C), stride)
    #         # [B, D, num_clips_group]
            
    #         y = x.new_zeros(B, D, self.C, num_clips_out)
    #         offset = (y_group.shape[-1] - num_clips_out) / (self.C - 1)
    #         for c in range(self.C):
    #             s = int(offset * c)
    #             e = int(offset * c + num_clips_out)
    #             y[:,:,c] = y_group[:,:,s:e]
    #         map2d[:, :, :, i, j] = y
    #     return map2d