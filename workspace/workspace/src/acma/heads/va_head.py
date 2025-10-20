# -*- coding: utf-8 -*-
"""
VA 轨迹回归头占位：输出 T×2 的 Valence/Arousal
"""
from typing import Any, Dict
import torch.nn as nn
import torch
import torch.nn.functional as F
from acma.gating.consistency import AVConsistencyGates

class VAHead(nn.Module):
    def __init__(self, in_features, num_class=0):
        super(VAHead, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features * 4, 2)
        self.num_class = num_class
        self.fc3 = nn.Linear(in_features * 4, self.num_class)
        self.av_gate = AVConsistencyGates()
        self.raw_beta = nn.Parameter(torch.tensor(0.0))

    def VA_calu(self, x):
        x = torch.tanh(self.fc2(self.act(self.fc1(x))))

        return x
    
    def emo_classification(self, x, av_mod, vod_mod, Tau=1.0):
        logits = self.fc3(self.act(self.fc1(x)))
        av_coh = self.av_gate(av_mod, vod_mod)  # [batch_size, timestep]
        beta = F.softplus(self.raw_beta)
        scores = -beta * av_coh
        w = torch.softmax(scores, dim=1)
        clip_logits = (w.unsqueeze(-1) * logits).sum(dim=1)  # Clip-Level 

        return F.softmax(clip_logits/Tau, dim=-1)



    def forward(self, x, av_mod=None, vod_mod=None):
        
        VA_pre = self.VA_calu(x)

        # if self.num_class != 0:

        #     clf_result = self.emo_classification(x, av_mod, vod_mod, Tau=1.0)
        
        # else: 
        #     clf_result = None

        # return [VA_pre, clf_result]
        return VA_pre
    

if __name__ == "__main__":

    v1 = torch.randn(50, 32, 128)  
    v2 = torch.randn(50, 32, 128)

    VAHead()






        


    
