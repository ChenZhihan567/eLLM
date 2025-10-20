# -*- coding: utf-8 -*-
"""
一致性门占位：Gate1/2 + 温度裁剪
"""
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class AVConsistencyGates(nn.Module):

    """AV confidence level"""

    def __init__(self):
        super(AVConsistencyGates, self).__init__()
        self.raw_alpha = nn.Parameter(torch.zeros(1)).to("cuda")
        self.raw_beta = nn.Parameter(torch.zeros(1)).to("cuda")


    def gate_av(self, av_mod, vid_mod):
        cos = F.cosine_similarity(av_mod, vid_mod, dim=-1, eps=1e-8)
        d = ((1.0 - cos) * 0.5).clamp(0.0, 1.0)   # Disagree rate in [0, 1]
        return d
    

    def forward(self, av_mod, vid_mod):

        alpha = F.softplus(self.raw_alpha).to(device=av_mod.device, dtype=av_mod.dtype)  # Smooth extent                  
        beta  = torch.sigmoid(self.raw_beta).to(device=av_mod.device, dtype=av_mod.dtype)  # Tolerance extent
        d = self.gate_av(av_mod, vid_mod)
        return torch.sigmoid(alpha * (beta - d)) 



def TextAVConsistencyGates(agree_av, VA_pre, VA_txt, tau=1.0):
    

    w = torch.softmax(agree_av, dim=1)
    clip_VApre = (w.unsqueeze(-1) * VA_pre).sum(dim=1)
    clip_VApre = F.layer_norm(clip_VApre, (clip_VApre.size(-1),))

    cos = F.cosine_similarity(clip_VApre, VA_txt, dim=-1, eps=1e-8)

    return F.sigmoid(tau * cos)






    



if __name__ == "__main__":
    
    avmodel = AVConsistencyGates()

    v1 = torch.randn(50, 32, 128).to("cuda")  # (batch=50, timestep = 32, dim=128)
    v2 = torch.randn(50, 32, 128).to("cuda")
    print(v1.device)
    print(v2.device)

    print(avmodel(v1, v2).shape)


