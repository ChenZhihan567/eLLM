# video_vediomae.py
import numpy as np
import torch
import torch.nn as nn
from transformers import VideoMAEModel
from acma.preprocessing.data_loader import dataloader
from acma.encoders.video_loader import VideoLoaderCV
from acma.config import get_default_config
import torch.nn.functional as F


CFG = get_default_config()

def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters(): p.requires_grad = flag

def unfreeze_last_n_blocks(vmae: VideoMAEModel, n: int):
    if n <= 0: return
    enc = getattr(vmae, "encoder", None)
    if not (enc and hasattr(enc, "layer")): return
    L = len(enc.layer); n = min(n, L)
    for i, blk in enumerate(enc.layer):
        set_requires_grad(blk, i >= L - n)

class MicroLocal(nn.Module):
    def __init__(self, d=256, dropout=0.0, tau=0.25):
        super().__init__()
        self.tau = tau
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64,128,3,2,1),   nn.ReLU(True),
            nn.Conv2d(128,d,3,2,1),    nn.ReLU(True),
        )
        self.drop = nn.Dropout(dropout)
        self.attn = nn.Conv2d(d, 1, 1)
        self.ln   = nn.LayerNorm(d)
        self.ffn  = nn.Sequential(nn.Linear(d,4*d), nn.GELU(), nn.Linear(4*d,d))
    def forward(self, x_btchw):  # (B,T,3,H,W) in [0,1]
        B,T,C,H,W = x_btchw.shape
        f = self.stem(x_btchw.reshape(B*T,C,H,W))
        f = self.drop(f)
        a = self.attn(f).flatten(2) / max(self.tau,1e-6)       # (B*T,1,hw)
        a = torch.softmax(a, dim=-1)
        v = f.flatten(2).transpose(1,2)                        # (B*T,hw,d)
        pooled = torch.bmm(a, v).squeeze(1)                    # (B*T,d)
        z = self.ln(pooled); z = z + self.ffn(z)
        return z.view(B,T,-1)                                  # (B,T,d)

class FusionHead(nn.Module):
    def __init__(self, d_g, d_m, d_out, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.proj_g = nn.Linear(d_g, d_out)
        self.proj_m = nn.Linear(d_m, d_out)
        self.mix    = nn.Linear(2*d_out, d_out)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_out, nhead=n_heads,
            dim_feedforward=4*d_out, dropout=dropout,
            batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln  = nn.LayerNorm(d_out)
    def forward(self, G, M, key_padding_mask=None):            # (B,Tg,dg),(B,Tm,dm)
        T = min(G.shape[1], M.shape[1])
        G = self.proj_g(G[:, :T, :]); M = self.proj_m(M[:, :T, :])
        X = self.mix(torch.cat([G, M], dim=-1))
        X = self.enc(X, src_key_padding_mask=key_padding_mask)
        return self.ln(X)                                      # (B,T,d_out)

class VideoMAEEncoder(nn.Module):
    """
    输入:  (B,T,3,H,W) ∈ [0,1] 或 frames_np (T,H,W,3) → encode_frames
    输出:  (B,T,D_out)
    """
    def __init__(self, file_list, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = VideoMAEModel.from_pretrained(cfg.video.model_id)
        hidden = self.backbone.config.hidden_size


        # Progressive Freezing of VideoMAE
        if getattr(cfg.video, "freeze_backbone", True):
            set_requires_grad(self.backbone, False)
        unfreeze_last_n_blocks(self.backbone, getattr(cfg.video, "finetune_last_n_blocks", 0))

        self.video_load(file_list)
        self.target_frames = np.floor(np.array(self.duration) * self.cfg.hard_alignment.target_hz).astype(int)


        mic = cfg.video.micro
        self.micro  = MicroLocal(d=mic.d, dropout=mic.dropout, tau=mic.tau) if mic.enabled else None
        fus = cfg.video.fusion
        d_out = fus.dim
        d_m   = mic.d if self.micro is not None else hidden
        self.fusion = FusionHead(d_g=hidden, d_m=d_m, d_out=d_out,
                                 n_layers=fus.n_layers, n_heads=fus.n_heads, dropout=fus.dropout)
        self.out_dim = getattr(cfg.video.norm, "proj_dim", d_out)
        self.proj = nn.Identity() if self.out_dim == d_out else nn.Linear(d_out, self.out_dim)
        self.out_ln = nn.LayerNorm(self.out_dim) if getattr(cfg.video.norm, "use_ln", True) else nn.Identity()
        self.to(self.device)
        self.eval()

    # ---- helpers ----
    def _sliding_clips(self, T:int):
        win   = getattr(self.cfg.video, "clip_len", 16)
        stride= getattr(self.cfg.video, "stride", 8)
        idx, s = [], 0
        while s < T:
            e = s + win
            if e > T: e, s = T, max(0, T - win)
            idx.append((s, e))
            if e == T: break
            s += stride
        return idx

    def _expected_in_channels(self)->int:
        pe = getattr(getattr(self.backbone, "embeddings", None), "patch_embeddings", None)
        if pe is not None:
            proj = getattr(pe, "projection", None) or getattr(pe, "proj", None)
            if proj is not None and hasattr(proj, "in_channels"): return int(proj.in_channels)
            if hasattr(pe, "num_channels"): return int(pe.num_channels)
        return int(getattr(self.backbone.config, "num_channels", 3))

    def _normalize(self, x: torch.Tensor, axis:int)->torch.Tensor:
        shape = [1,1,1,1,1]; shape[axis]=3
        mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(*shape)
        std  = torch.tensor([0.229,0.224,0.225], device=x.device).view(*shape)
        return (x - mean) / std

    def _ensure_channels_axis(self, x: torch.Tensor, axis:int, exp:int)->torch.Tensor:
        C = x.shape[axis]
        if C == exp: return x
        if axis==1:   # BCTHW
            if exp==1 and C==3:
                y = 0.2989*x[:,0:1]+0.5870*x[:,1:2]+0.1140*x[:,2:3]; return y
            if exp==3 and C==1: return x.repeat(1,3,1,1,1)
        if axis==2:   # BTCHW
            if exp==1 and C==3:
                y = 0.2989*x[:,:,0:1]+0.5870*x[:,:,1:2]+0.1140*x[:,:,2:3]; return y
            if exp==3 and C==1: return x.repeat(1,1,3,1,1)
        raise ValueError(f"[ensure_channels] got C={C}, expected {exp}, axis={axis}, shape={tuple(x.shape)}")

    def _encode_clip_global_adaptive(self, clip_btchw: torch.Tensor)->torch.Tensor:
        exp_c = self._expected_in_channels()
        # Try BTCHW
        xA = self._normalize(clip_btchw, axis=2)
        xA = self._ensure_channels_axis(xA, axis=2, exp=exp_c)
        try:
            out = self.backbone(pixel_values=xA); return out.last_hidden_state.mean(dim=1)
        except Exception as eA:
            # Try BCTHW
            xB = clip_btchw.permute(0,2,1,3,4).contiguous()
            xB = self._normalize(xB, axis=1)
            xB = self._ensure_channels_axis(xB, axis=1, exp=exp_c)
            try:
                out = self.backbone(pixel_values=xB); return out.last_hidden_state.mean(dim=1)
            except Exception as eB:
                raise RuntimeError(f"backbone refused both layouts | BTCHW: {eA} | BCTHW: {eB}")

    @staticmethod
    def _interp_clip_series_to_frames(clip_mat: torch.Tensor, T:int, centers:list)->torch.Tensor:
        hid = int(clip_mat.shape[1])
        clip_np = clip_mat.detach().float().cpu().numpy()
        centers = np.asarray(centers, dtype=np.float32)
        frame_pos = np.arange(T, dtype=np.float32) + 0.5
        frame_g = np.zeros((T, hid), dtype=np.float32)
        for d in range(hid):
            frame_g[:,d] = np.interp(frame_pos, centers, clip_np[:,d], left=clip_np[0,d], right=clip_np[-1,d])
        return torch.from_numpy(frame_g).unsqueeze(0)  # (1,T,hid)

    # ---- core forward ----
    def _encode_one(self, x_btchw: torch.Tensor)->torch.Tensor:
        # x_btchw: (1,T,3,H,W) in [0,1]
        T = x_btchw.shape[1]
        M = self.micro(x_btchw) if self.micro is not None else None

        wins = self._sliding_clips(T)
        
        vecs, centers = [], []
        step = getattr(self.cfg.video, "clip_len", 16)
        for (s,e) in wins:
            clip = x_btchw[:, s:e]
            if clip.shape[1] < step:
                pad = clip[:, -1:].repeat(1, step-clip.shape[1], 1, 1, 1)
                clip = torch.cat([clip, pad], dim=1)
            g_clip = self._encode_clip_global_adaptive(clip)  # (1, hidden)
            vecs.append(g_clip); centers.append((s+e)/2.0)
        clip_mat = torch.cat(vecs, dim=0).to(self.device)
        G = self._interp_clip_series_to_frames(clip_mat, T, centers).to(self.device)  # (1,T,hidden)
        if M is None: M = G
        Y = self.fusion(G, M, key_padding_mask=None)  # (1,T,d_out)
        Y = self.proj(Y); Y = self.out_ln(Y)
        return Y

    def forward(self):

        B = self.x_btchw.shape[0]

        # x_btchw: (B,T,3,H,W) in [0,1]
        self.x_btchw = self.x_btchw.to(self.device)

        # outs = [self.linear_interpolation(self._encode_one(self.x_btchw[b:b+1]), self.target_frame) for b in range(self.x_btchw.shape[0])]

        aligned_list = []

        # Linearly interpolate each sample to its respective frame length
        for i, tf in enumerate(self.target_frames):
            x = self.x_btchw[i:i+1]
            x = self._encode_one(x)
            aligned_i = self.linear_interpolation(x, tf)
            aligned_list.append(aligned_i)

        aligned = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in aligned_list], batch_first=True)

        max_tar_frames = max(self.target_frames)


        mask = torch.zeros((B, max_tar_frames), dtype=torch.bool, device=self.cfg.audio.device)
        for b in range(B):
            mask[b, :self.target_frames[b]] = True


        # return torch.cat(outs, dim=0)  # (B,T,D)
        return aligned, mask
    

    @torch.no_grad()
    def encode_frames(self, frames_np: np.ndarray):
        # frames_np: (T,H,W,3) uint8
        T = int(frames_np.shape[0]); d = self.out_dim
        if T == 0:
            empty = torch.zeros((1,0,d), device=self.device, dtype=torch.float32)
            return empty, {"frames":0, "dim":d, "device":str(self.device)}
        x = torch.from_numpy(frames_np).to(self.device, dtype=torch.float32)/255.0
        x = x.permute(0,3,1,2).unsqueeze(0)  # (1,T,3,H,W)
        feats = self._encode_one(x)
        return feats, {"frames":T, "dim":feats.shape[-1], "device":self.device.type}
    
    def video_load(self, file_list):
        vl = VideoLoaderCV(file_list, self.cfg)
        self.x_btchw, self.duration = vl.load_all_videos()
        return 


    def linear_interpolation(self, hidden_states, target_frame):

        # Rearrange to (B, D, T) for torch.interpolate compatibility
        x = hidden_states.permute(0, 2, 1)

        # Perform linear interpolation (align_corners=False ensures proportional scaling)
        aligned = F.interpolate(x, size=target_frame, mode="linear", align_corners=False)
        
        # Restore original order: (B, T, D)
        aligned = aligned.permute(0, 2, 1) 
        return aligned

    











if __name__ == "__main__":

    dl = dataloader()

    for idx, batch in enumerate(dl):
        vme = VideoMAEEncoder(batch)
        encoding, mask = vme()
        print(f"Video encoding {idx}: ", encoding.shape)

   
