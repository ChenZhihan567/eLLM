# train_video_batch.py
import os, torch, torch.nn as nn
from typing import List
from config import get_default_config
from video_mae import VideoMAEEncoder
from video_loader import make_loader
from utils_align import to_len_interpolate
from labels_utils import load_labels_file, labels_to_indices   # ← 新增

TARGET_LEN = 25
USE_LABEL_COLUMN = "emotion"   # 'emotion' or 'polarity'
NUM_CLASSES = None             # 由标签文件自动推断

def gather_video_paths(root: str) -> List[str]:
    return sorted([os.path.join(root,f) for f in os.listdir(root)
                   if f.lower().endswith((".mp4",".avi",".mov",".mkv"))])





def main():
    cfg = get_default_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 路径列表（顺序用于与标签对齐；有了 idx 就不强依赖顺序了）
    video_paths = gather_video_paths(cfg.dataloader.path)

    # 2) 读标签文件（按情绪/极性列）
    labels_txt = os.path.join(cfg.dataloader.path, "labels.txt")
    rows, class_to_idx = load_labels_file(labels_txt, use_column=USE_LABEL_COLUMN)
    target_indices_full = torch.tensor(labels_to_indices(rows, class_to_idx), dtype=torch.long)
    assert len(target_indices_full) >= len(video_paths), \
        f"标签数({len(target_indices_full)})少于视频数({len(video_paths)})，请检查对齐。"
    global NUM_CLASSES
    NUM_CLASSES = len(class_to_idx)

    # 3) DataLoader
    loader = make_loader(video_paths, cfg, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0)

    # 4) 模型 & 任务头
    video = VideoMAEEncoder(cfg).to(device); video.train()
    in_dim = getattr(cfg.video.norm, "proj_dim", cfg.video.fusion.dim)
    head  = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, NUM_CLASSES)).to(device)

    params = [p for p in video.parameters() if p.requires_grad] + list(head.parameters())
    optim = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    crit  = nn.CrossEntropyLoss()

    for epoch in range(3):
        for x_batch, lengths, paths, idxs in loader:   # ← 接收 idxs
            x_batch = x_batch.to(device)
            feats = video(x_batch)                     # (B,T,D)
            feats25, _ = to_len_interpolate(feats, TARGET_LEN)
            logits = head(feats25.mean(dim=1))         # (B, num_classes)

            # 用 idxs 从“全体标签索引”里拿出当前 batch 的标签
            y = target_indices_full[idxs].to(device)   # (B,)

            loss = crit(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward(); optim.step()

        print(f"epoch {epoch+1}: loss={loss.item():.4f}")

if __name__ == "__main__":
    main()
