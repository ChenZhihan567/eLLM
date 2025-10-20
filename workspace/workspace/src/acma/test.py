import os
import ffmpeg
import numpy as np
from acma.config import get_default_config

CFG = get_default_config()


class VideoLoaderCV:
    def __init__(self, cfg=None, file_list=None):
        self.cfg = cfg
        self.path = cfg.dataloader.path
        self.h, self.w = map(int, getattr(cfg.video, "target_size", (224, 224)))

        self.file_list = file_list

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

    def load_frames(self):

        proc = (
            ffmpeg
            .input()
        )

    
