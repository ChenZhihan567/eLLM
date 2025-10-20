# encoders/__init__.py

"""
视频编码器工厂模块

提供统一接口创建不同的视频编码器
"""

from acma.config import get_default_config


def create_video_encoder(file_list, cfg=None):
    """
    视频编码器工厂函数

    根据cfg.video.backbone自动选择编码器

    Args:
        file_list: List[str] - mp4文件路径列表
        cfg: SimpleNamespace - 配置对象

    Returns:
        encoder: 视频编码器实例
            - 输入: file_list
            - 输出: forward() -> (features, mask)
                features: (B, T, 256)
                mask: (B, T) bool

    Example:
        >>> cfg = get_default_config()
        >>> cfg.video.backbone = "timesformer"
        >>> encoder = create_video_encoder(['video1.mp4'], cfg)
        >>> features, mask = encoder()
    """
    if cfg is None:
        cfg = get_default_config()

    backbone = cfg.video.backbone.lower()

    if backbone == "videomae":
        from acma.encoders.video_mae import VideoMAEEncoder
        print(f"[Factory] Using VideoMAE encoder")
        return VideoMAEEncoder(file_list, cfg)

    elif backbone == "timesformer":
        try:
            from acma.encoders.video_timesformer import VideoTimeSformerEncoder
            print(f"[Factory] Using TimeSformer encoder")
            return VideoTimeSformerEncoder(file_list, cfg)
        except ImportError as e:
            raise ImportError(
                f"TimeSformer encoder not available. "
                f"Make sure video_timesformer.py exists. Error: {e}"
            )

    elif backbone == "slowfast":
        try:
            from acma.encoders.video_slowfast import VideoSlowFastEncoder
            print(f"[Factory] Using SlowFast encoder")
            return VideoSlowFastEncoder(file_list, cfg)
        except ImportError as e:
            raise ImportError(
                f"SlowFast encoder not available. "
                f"Install with: pip install pytorchvideo fvcore. Error: {e}"
            )

    else:
        raise ValueError(
            f"Unknown video backbone: '{backbone}'. "
            f"Available options: 'videomae', 'timesformer', 'slowfast'"
        )