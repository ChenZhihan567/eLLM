"""
Cross-Attention 融合（對稱 A↔V）
回傳:
  F_t, A2V, V2A 皆為 (B,T,d_model)
"""
import torch
import torch.nn as nn
from acma.encoders.audio_whisper import AudioWhisperEncoder
from acma.encoders.video_mae import VideoMAEEncoder
from acma.gating.consistency import AVConsistencyGates
from acma.config import get_default_config

CFG = get_default_config()

def causal_mask(q, k):

    T1 = q.shape[1]
    T2 = k.shape[1]

    mask = torch.triu(torch.ones(T1, T2, dtype=torch.bool), diagonal=1).to(q.device)
    return mask

def mask_zero(x, valid_mask):
    valid_mask.to(dtype=torch.bool, device=x.device)
    return x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


CFG = get_default_config()

class HiddenAligner(nn.Module):
    def __init__(self, tar_dim=768):
        super().__init__()

        self.fc1 = nn.Linear(CFG.audio.dim, tar_dim)
        self.fc2 = nn.Linear(CFG.video.dim, tar_dim)

    def forward(self, audio_x, video_x, audio_mask=None, video_mask=None):
        naudio_x = self.fc1(audio_x)
        nvideo_x = self.fc2(video_x)

        if audio_mask is not None:
            naudio_x = mask_zero(naudio_x, audio_mask)
        if video_mask is not None:
            nvideo_x = mask_zero(nvideo_x, video_mask)

        return naudio_x, nvideo_x




class CrossModalAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 768, nhead: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.mha_vq_a = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mha_aq_v = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ln_v1 = nn.LayerNorm(d_model)
        self.ln_a1 = nn.LayerNorm(d_model)
        self.ln_v2 = nn.LayerNorm(d_model)
        self.ln_a2 = nn.LayerNorm(d_model)

        hidden = int(d_model * mlp_ratio)
        self.ffn_v = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_model))
        self.ffn_a = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_model))
        self.do = nn.Dropout(dropout)

    def forward(self, A, V, A_mask, V_mask, w_t: torch.Tensor | None = None):
        # (B,T,D) -> (T,B,D) for MHA

        # V<-A

        V_atten_mask = causal_mask(V, A)
        print("cas_mask: ", V_atten_mask.shape)

        Vq, A_kv = self.ln_v1(V), self.ln_a1(A)
        V2A, _ = self.mha_vq_a(query=Vq, key=A_kv, value=A_kv, need_weights=False, key_padding_mask=~A_mask.bool(), attn_mask=V_atten_mask)
        V2A = mask_zero(V2A, V_mask)

        V2A = V + self.do(V2A)
        V2A = mask_zero(V2A, V_mask)

        V2A = V2A + self.do(self.ffn_v(self.ln_v2(V2A)))
        V2A = mask_zero(V2A, V_mask)

        #2) 前饋網路（FNN / FFN）

        # A<-V

        A_atten_mask = causal_mask(A, V)
        print("cas_mask: ", A_atten_mask.shape)

        Aq, V_kv = self.ln_a1(A), self.ln_v1(V)
        A2V, _ = self.mha_aq_v(query=Aq, key=V_kv, value=V_kv, need_weights=False, key_padding_mask=~V_mask.bool(), attn_mask=A_atten_mask)
        A2V = mask_zero(A2V, A_mask)

        A2V = A + self.do(A2V)
        A2V = mask_zero(A2V, A_mask)

        A2V = A2V + self.do(self.ffn_a(self.ln_a2(A2V)))
        A2V = mask_zero(A2V, A_mask)
        #2) 前饋網路（FNN / FFN）

        # # back to (B,T,D) 3) 轉回 batch-first
   
        print(V2A.shape)
        print(A2V.shape)

        # # 融合（先不加 gate；之後接上 video/gate 再餵 w_t）4) 融合成 F_t（支援步級權重 w_t）
        # #w_t 由「Gate 1：A/V 一致性」產生（如：w_t = softmax(-β·(1 - cos(A_t,V_t)))
        # #w_t 越大 → 更相信 V2A（視覺導向音訊）
        # #1 - w_t 越大 → 更相信 A2V（音訊導向視覺）
        # #沒有 w_t 時，用簡單的平均當作融合。
        gate1 = AVConsistencyGates()
        w_t = gate1(A2V, V2A)

        if w_t is not None:
            wt = w_t.clamp(0, 1).unsqueeze(-1)  # (B,T,1)
            F_t = wt * V2A + (1.0 - wt) * A2V
        else:
            F_t = 0.5 * (V2A + A2V)

        return F_t, A2V, V2A
    



if __name__ == "__main__":

    pass


