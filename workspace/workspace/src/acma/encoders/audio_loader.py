"""
audio_loader.py
加载给定的 mp4 文件列表，提取音频为 Whisper 可用格式
"""

import os
import numpy as np
import ffmpeg
from acma.config import get_default_config
from acma.preprocessing.data_loader import dataloader
from moviepy.editor import VideoFileClip


CFG = get_default_config()

class AudioLoader:
    """音频加载器，支持传入 mp4 文件列表"""

    def __init__(self, file_list=None, cfg=CFG):
        """
        Args:
            cfg: 全局配置
            file_list (list[str]): mp4 文件路径列表
        """
        self.cfg = cfg if cfg is not None else get_default_config()
        self.target_sr = self.cfg.audio.target_sr

        # 如果传入了文件列表就用，否则默认取 config.data.path
        #if file_list is not None:
        self.file_list = file_list
        #else:
            # 如果 config.data.path 是字符串，就用它
            #if isinstance(self.cfg.dataloader.path, str):
                #self.file_list = [self.cfg.dataloader.path]
            #elif isinstance(self.cfg.dataloader.path, list):
                #self.file_list = self.cfg.dataloader.path
            #else:
                #raise ValueError("cfg.dataloader.path 必须是字符串或字符串列表")

        base_dir = getattr(self.cfg, "dataloader", None)
        base_dir = getattr(base_dir, "path", None)

        fixed_list = []
        for f in self.file_list:
            if not os.path.isabs(f) and base_dir is not None:
                f = os.path.join(base_dir, f)
            if not os.path.isfile(f):
                raise FileNotFoundError(f"文件不存在: {f}")
            fixed_list.append(f)
        self.file_list = fixed_list


    def load_audio_from_mp4(self, file_path):
        """从单个 mp4 提取音频，返回 numpy array"""
        try:
            proc = (
                ffmpeg
                .input(file_path)
                .output(
                    'pipe:',
                    format='f32le',
                    ac=1,
                    ar=self.target_sr,
                    loglevel='error'
                )
                .run(capture_stdout=True, capture_stderr=True)
            )
            audio_bytes = proc[0]
            if not audio_bytes:
                raise RuntimeError(f"ffmpeg 未返回音频数据: {file_path}")

            return np.frombuffer(audio_bytes, dtype=np.float32)
          
        except Exception as e:
            raise RuntimeError(f"解码失败 {file_path}: {e}")

    def load_all_audios(self):
        """加载文件列表中的所有 mp4 文件音频"""
        data = []
        duration = []
        for fp in self.file_list:
            audio_np = self.load_audio_from_mp4(fp)
            # data.append((fp, audio_np))
            data.append(audio_np)

            clip = VideoFileClip(fp)

            durtime = float(clip.duration) if clip.duration is not None else 0.0
            duration.append(durtime)
    
        return data, duration


if __name__ == "__main__":

    

    al = AudioLoader(CFG, ['dia2_utt1.mp4', 'dia1_utt15.mp4', 'dia1_utt0.mp4', 'dia1_utt4.mp4'])
    
    a = al.load_all_audios()
    for w in a:
            print(np.max(np.abs(w)))
    
