"""Ouput: (B, T, C, H, W)
"""

# video_loader.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from acma.config import get_default_config
from acma.preprocessing.data_loader import dataloader
from moviepy.editor import VideoFileClip
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="moviepy.*")



CFG = get_default_config()


# ---------------------------------------------------------
# 单视频帧加载器
# ---------------------------------------------------------
class VideoLoaderCV:
    """
    - 使用 cfg.video.target_size 控制输入分辨率
    - 支持时间下采样: cfg.video.frame_stride
    - 支持最大帧限制: cfg.video.max_frames
    输出: np.ndarray (T, H, W, 3), RGB, uint8
    """
    def __init__(self, file_list, cfg=CFG):
        self.cfg = cfg
        self.h, self.w = map(int, getattr(self.cfg.video, "target_size", (224, 224)))

        base_dir = getattr(self.cfg, "dataloader", None)
        base_dir = getattr(base_dir, "path", None)

        fixed_list = []
        for f in file_list:
            if not os.path.isabs(f) and base_dir is not None:
                f = os.path.join(base_dir, f)
            if not os.path.isfile(f):
                raise FileNotFoundError(f"文件不存在: {f}")
            fixed_list.append(f)
        self.paths = fixed_list


    @staticmethod
    def _apply_orientation(img, orientation: int):
        if orientation == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if orientation == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        if orientation == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def load_frames(self, file_path: str) -> np.ndarray:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)

        cap = cv2.VideoCapture(file_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(file_path, cv2.CAP_ANY)
        if not cap.isOpened():
            raise RuntimeError(f"[VideoLoaderCV] Cannot open video: {file_path}")

        orientation = 0
        if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
            try:
                val = cap.get(cv2.CAP_PROP_ORIENTATION_META)
                if isinstance(val, (int, float)) and not np.isnan(val):
                    orientation = int(val) % 360
            except Exception:
                orientation = 0

        # ↓ 新增：时间下采样与最大帧限制（显存友好）
        frame_stride = int(getattr(self.cfg.video, "frame_stride", 1)) or 1
        max_frames = getattr(self.cfg.video, "max_frames", None)
        frames = []
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            if (i % frame_stride) != 0:
                i += 1
                continue
            i += 1

            if bgr is None:
                continue

            # 灰度 → BGR
            if bgr.ndim == 2 or (bgr.ndim == 3 and bgr.shape[2] == 1):
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

            if orientation:
                bgr = self._apply_orientation(bgr, orientation)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_AREA)
            frames.append(rgb)

            if (max_frames is not None) and (len(frames) >= int(max_frames)):
                break

        cap.release()

        if not frames:
            return np.zeros((0, self.h, self.w, 3), dtype=np.uint8)
        arr = np.stack(frames, axis=0).astype(np.uint8, copy=False)
        return np.ascontiguousarray(arr)

# ---------------------------------------------------------
# DataLoader 封装：批量视频加载 & 补齐/截断
# ---------------------------------------------------------

    def load_all_videos(self):
        """加载文件列表中的所有 mp4 视频音频"""
        data = []
        duration = []
        for fp in self.paths:
            # print(fp)
            frames = self.load_frames(fp) 
            x = torch.from_numpy(frames).float() / 255.0    # (T, H, W, 3)
            x = x.permute(0, 3, 1, 2).contiguous() 
            # data.append((fp, audio_np))
            data.append(x)

            clip = VideoFileClip(fp)

            durtime = float(clip.duration) if clip.duration is not None else 0.0
            duration.append(durtime)
    
        return self.collate_video_batch(data), duration
        # return data

    
    
    def collate_video_batch(self, batch):
        # batch: List[(T_i, 3, H, W), path, idx]
        # xs, paths, idxs = zip(*batch)
        xs = batch
        # print(xs)
        lengths = torch.tensor([t.shape[0] for t in xs], dtype=torch.long)
        # print(lengths)
        T_max = int(lengths.max())
        # print(T_max)

        cap_T = getattr(getattr(self.cfg, "video", {}), "max_frames", None)
        if cap_T is not None:
            T_max = min(T_max, int(cap_T))
        # print(T_max)
        padded = []
        for x in xs:
            # 截断到 T_max
            if x.shape[0] > T_max:
                x = x[:T_max]
            # 末帧补齐
            if x.shape[0] < T_max:
                pad = x[-1:].repeat(T_max - x.shape[0], 1, 1, 1)
                x = torch.cat([x, pad], dim=0)
            # print(x.shape)
            padded.append(x)

       

        x_batch = torch.stack(padded, dim=0)  # (B, T_max, 3, H, W)
        lengths_c = lengths.clamp_max(T_max)
        # print(lengths_c)
        # return x_batch, lengths_c, list(paths), torch.tensor(idxs, dtype=torch.long)
        return x_batch



if __name__ == "__main__":
    # dl = make_loader(['dia1_utt6.mp4', 'dia1_utt9.mp4', 'dia1_utt14.mp4', 'dia1_utt13.mp4'], CFG)

    dl = dataloader()

    for i in dl:
        vcv = VideoLoaderCV(i, CFG)
        a, d = vcv.load_all_videos()
        print(d)




