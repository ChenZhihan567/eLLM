import torch
import torch.nn as nn
import torch.nn.functional as F
from acma.corrector.direction import MoERouter
from acma.corrector.strength import Strength
from acma.config import get_default_config


class Corrector:
    """
        This class is responsible for correcting the Valence-Arousal values.
    """    
    def __init__(self, f_t, t_t, coh_av, coh_txt, VA_txt, VApre):
        self.f_t = f_t  # [batch_size, frame, hidden_size]
        self.t_t = t_t
        self.coh_av = coh_av  # [batch_size, frame]
        self.coh_txt = coh_txt
        self.VA_txt = VA_txt
        self.VApre = VApre
        self.cfg = get_default_config()
        self.clipLevelUnifyFeature()

    def clipLevelUnifyFeature(self):

        # AV_fusion Clip-Level 
        w = torch.softmax(self.coh_av, dim=1)
        clip_ft = (w.unsqueeze(-1) * self.f_t).sum(dim=1)
        self.clip_ft = F.layer_norm(clip_ft, (clip_ft.size(-1),))  # [batch_size, cat_hidden_size]

        # Sentence Clip-Level
        w = torch.softmax(self.coh_av, dim=1)
        clip_VApre = (w.unsqueeze(-1) * self.VApre).sum(dim=1)
        self.clip_VApre = F.layer_norm(clip_VApre, (clip_VApre.size(-1),))


        # print("clipf_t: ", clip_ft.shape)
        # print("txt: ", self.t_t.shape)
        # print("coh_av", self.coh_av.shape)
        # print("coh_txt", self.coh_txt.shape)
        # print("VA_txt: ", self.VA_txt.shape)
        # print("clip_VA_pre:", clip_VApre.shape)


        C = torch.cat([clip_ft, self.t_t, self.coh_av, self.coh_txt, self.VA_txt, clip_VApre], dim=-1).to("cuda")
        self.C = F.layer_norm(C, (C.size(-1),))  # [batch_size, cat_hidden_size]

        print("C shape: ", self.C.shape)

        return self.C
    
    
    def correcting(self):
        director = MoERouter(self.cfg.corrector.num_experts, self.C.shape[-1] , self.C.shape[-1] * 4)
        delta = director(self.C, tau=1.0)
        print("direction: ", delta.shape)

        strengthor = Strength(self.C.shape[-1], self.coh_av, self.coh_txt)
        lam = strengthor(self.C, phi=1.0)
        print("strength: ", lam.shape)

        VApost = self.clip_VApre + delta * lam
        VApost = torch.clamp(VApost, min=-1.0, max=1.0)
        print("VA_post:", VApost.shape)
        return VApost



        

    


if __name__ == "__main__":
    input_size = 5
    hidden_size = 10
    num_experts = 3
    batch_size = 10

    model = Strength(input_size, 0.6, -0.1)

    # model = Expert(input_size, hidden_size)
    demo = torch.randn(batch_size, input_size)

    delata = model(demo)
    print(delata)




