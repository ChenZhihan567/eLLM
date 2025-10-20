import torch
import torch.nn as nn
import torch.nn.functional as F
from acma.corrector.direction import MoERouter


class Strength(nn.Module):
    
    def __init__(self, in_features, coh_av, coh_txt):
        super(Strength, self).__init__()
        self.fc1 = nn.Linear(in_features, 1).to("cuda")
        self.coh_av = coh_av
        self.coh_txt = coh_txt
        self.alpha_raw = nn.Parameter(torch.zeros(1)).to("cuda")
        self.w_mod_raw = nn.Parameter(torch.zeros(2)).to("cuda")


    def forward(self, x, phi=1.0):

        lam_raw = F.sigmoid(self.fc1(x))  # (batch_size, 1)

        w_mod = F.softmax(self.w_mod_raw, dim=0)
        # print(w_mod)
        alpha = F.sigmoid(self.alpha_raw)
        # print(alpha)
        beta = 1 - alpha
        # print(beta)

        # print("alpha: ", alpha.shape)
        # print("beta: ", beta.shape)
        # print("w_av: ", w_av.shape)
        # print("coh_av: ", self.coh_av.shape)
        # print("w_txt: ", w_txt.shape)
        # print("coh_txt: ", self.coh_txt.shape)


        coh_av_pooled = self.coh_av.mean(dim=1, keepdim=True)  # [4, 1]


        w_av = w_mod[0]
        w_txt = w_mod[1]
        coh_level = alpha + beta * (w_av * coh_av_pooled+ w_txt * self.coh_txt)
        print(coh_level)

        lam = phi * coh_level * lam_raw

        return lam


# def direction(x, num_experts, in_features, hidden_dim, tau=1.0):
#     model = MoERouter(num_experts, in_features, hidden_dim)
#     delta = model(x, tau)
#     return delta

# def strength(x, in_features, coh_av, coh_txt, phi=1.0):
#     model = Strength(in_features, coh_av, coh_txt)
#     lam = model(x)
#     return lam
    


