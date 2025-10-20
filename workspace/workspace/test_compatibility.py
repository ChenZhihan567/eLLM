# 测试脚本: test_compatibility.py

import torch
from acma.config import get_default_config
from acma.encoders.video_mae import VideoMAEEncoder
from acma.encoders.video_timesformer import VideoTimeSformerEncoder


def test_dimension_compatibility():
    """验证维度兼容性"""

    cfg = get_default_config()
    file_list = ['dia1_utt0.mp4', 'dia1_utt4.mp4']

    print("=" * 60)
    print("Testing Dimension Compatibility")
    print("=" * 60)

    # 测试VideoMAE
    print("\n[1] VideoMAE:")
    cfg.video.backbone = "videomae"
    encoder_mae = VideoMAEEncoder(file_list, cfg)
    feat_mae, mask_mae = encoder_mae()
    print(f"  Features: {feat_mae.shape}")
    print(f"  Mask: {mask_mae.shape}")
    print(f"  Mask dtype: {mask_mae.dtype}")

    # 测试TimeSformer
    print("\n[2] TimeSformer:")
    cfg.video.backbone = "timesformer"
    encoder_ts = VideoTimeSformerEncoder(file_list, cfg)
    feat_ts, mask_ts = encoder_ts()
    print(f"  Features: {feat_ts.shape}")
    print(f"  Mask: {mask_ts.shape}")
    print(f"  Mask dtype: {mask_ts.dtype}")

    # 验证
    print("\n[3] Compatibility Check:")
    assert feat_mae.shape == feat_ts.shape, "❌ Feature shape mismatch!"
    assert mask_mae.shape == mask_ts.shape, "❌ Mask shape mismatch!"
    assert mask_mae.dtype == mask_ts.dtype, "❌ Mask dtype mismatch!"
    assert feat_mae.shape[-1] == 256, "❌ Output dim should be 256!"

    print("  ✅ All dimensions match!")
    print("  ✅ VideoMAE and TimeSformer are compatible!")

    return True


if __name__ == "__main__":
    test_dimension_compatibility()