# src/acma/pipeline.py

from acma.preprocessing.data_loader import dataloader
from acma.encoders import create_video_encoder
from acma.encoders.audio_whisper import AudioWhisperEncoder
from acma.encoders.text_encoder import TextEncoder
from acma.fusion.cross_fusion import HiddenAligner, CrossModalAttentionBlock
from acma.heads.va_head import VAHead
from acma.gating.consistency import AVConsistencyGates, TextAVConsistencyGates
from acma.config import get_default_config
from acma.corrector.corrector import Corrector
import argparse
import os


class ACMAPipeline:

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg else get_default_config()
        self.device = self.cfg.device
        self.data_loader = dataloader()

        print(f"\n{'=' * 70}")
        print(f"ACMA Pipeline Initialized")
        print(f"{'=' * 70}")
        print(f"Device: {self.device}")
        print(f"Video Backbone: {self.cfg.video.backbone}")
        print(f"Batch Size: {self.cfg.train.batch_size}")
        print(f"{'=' * 70}\n")

    def run(self):
        for idx, batch in enumerate(self.data_loader):
            print(f"\n{'=' * 70}")
            print(f"Processing Batch {idx}")
            print(f"Videos: {batch}")
            print(f"{'=' * 70}\n")

            # 音频编码
            print(f"[1/8] Audio encoding...")
            audio_encoder = AudioWhisperEncoder(batch, self.cfg)
            audio_x, audio_mask = audio_encoder.extract_features()
            print(f"  ✓ Audio: {audio_x.shape}")

            # 视频编码
            print(f"\n[2/8] Video encoding ({self.cfg.video.backbone})...")
            video_encoder = create_video_encoder(batch, self.cfg)
            video_x, video_mask = video_encoder()
            print(f"  ✓ Video: {video_x.shape}")

            # 特征对齐
            print(f"\n[3/8] Feature alignment...")
            ha = HiddenAligner().to(self.device)
            naudio_x, nvideo_x = ha(audio_x, video_x, audio_mask, video_mask)
            naudio_x = naudio_x.to(self.device)
            nvideo_x = nvideo_x.to(self.device)
            audio_mask = audio_mask.to(self.device)
            video_mask = video_mask.to(self.device)
            print(f"  ✓ Aligned - Audio: {naudio_x.shape}, Video: {nvideo_x.shape}")

            # 交叉注意力
            print(f"\n[4/8] Cross-modal fusion...")
            mha = CrossModalAttentionBlock().to(self.device)
            F_t, A2V, V2A = mha(naudio_x, nvideo_x, audio_mask, video_mask)
            print(f"  ✓ Fusion: {F_t.shape}")

            # VA预测
            print(f"\n[5/8] VA prediction...")
            va_creator = VAHead(self.cfg.fusion.hidden_dim).to(self.device)
            VA_pre = va_creator(F_t)
            print(f"  ✓ VA_pre: {VA_pre.shape}")

            # 文本编码
            print(f"\n[6/8] Text encoding...")
            text_encoder = TextEncoder(self.cfg)
            text_x, VA_txt = text_encoder.extract_features(audio_encoder.audio_np)
            print(f"  ✓ Text: {text_x.shape}, VA_txt: {VA_txt.shape}")

            # 一致性门控
            print(f"\n[7/8] Consistency gating...")
            gate1 = AVConsistencyGates()
            agree_av = gate1(naudio_x, nvideo_x)
            print(f"  ✓ AV consistency: {agree_av.shape}")

            agree_txt = TextAVConsistencyGates(agree_av, VA_pre, VA_txt, tau=1.0)
            print(f"  ✓ Text consistency: {agree_txt.shape}")
            agree_txt = agree_txt.unsqueeze(-1)

            # VA修正
            print(f"\n[8/8] VA correction...")
            corr = Corrector(F_t, text_x, agree_av, agree_txt, VA_txt, VA_pre)
            VA_post = corr.correcting()
            print(f"  ✓ VA_post: {VA_post.shape}")

            print(f"\n{'=' * 70}")
            print(f"Batch {idx} completed!")
            print(f"{'=' * 70}\n")


def main():
    """主函数，支持命令行参数"""

    # 参数解析
    parser = argparse.ArgumentParser(description='ACMA Pipeline')
    parser.add_argument(
        '--backbone',
        type=str,
        choices=['videomae', 'timesformer', 'slowfast'],
        default=None,
        help='Video encoder backbone (default: from config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (default: from config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device: cuda:0 or cpu (default: auto)'
    )

    args = parser.parse_args()

    # 加载配置
    cfg = get_default_config()

    # 覆盖配置（如果提供了参数）
    if args.backbone:
        cfg.video.backbone = args.backbone
        print(f"[CLI] Using backbone: {args.backbone}")

    if args.batch_size:
        cfg.train.batch_size = args.batch_size
        print(f"[CLI] Using batch size: {args.batch_size}")

    if args.device:
        cfg.device = args.device
        cfg.audio.device = args.device
        cfg.text.device = args.device
        cfg.asr.device = args.device
        print(f"[CLI] Using device: {args.device}")

    # 运行pipeline
    pipeline = ACMAPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()