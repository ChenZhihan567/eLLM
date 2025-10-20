# config.py
"""This document is the configuration file for the entire project.
"""
from types import SimpleNamespace
from transformers import WhisperModel, WhisperFeatureExtractor
import torch

featurizer = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-base"
)
model = WhisperModel.from_pretrained("openai/whisper-base")


def get_default_config():
    """Generate the default configuration file for the project

    Returns:
        types.SimpleNamespace: Return the default configuration; you can later modify this file directly in VS Code to switch modules.
    """

    cfg = SimpleNamespace()

    # ============================================================
    # 🎯 新增: 全局设备配置 (必须在最前面)
    # ============================================================
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg.dataloader = SimpleNamespace(
        path="data/dev_splits_complete"
    )

    # cfg.model = SimpleNamespace(d_model = 256)

    cfg.audio = SimpleNamespace(
        dim=512,  # Whisper encoder output dimension
        target_sr=16000,  # I/O loading and usage
        whisper_id="openai/whisper-base",
        device=cfg.device,  # 🎯 使用全局设备
        freeze_encoder=True,  # Freeze the pre-trained encoder
    )

    cfg.hard_alignment = SimpleNamespace(
        target_hz=25
    )

    # ASR（转录）相关可调参数
    cfg.asr = SimpleNamespace(
        whisper_id=cfg.audio.whisper_id,  # 默认与 audio 一致，亦可改成别的 Whisper 重级
        language="en",  # 如 "en", "zh"
        task="transcribe",  # "transcribe" 或 "translate"
        fp16=True,  # 若在 CUDA 上且模型支持，开启半精度推理
        device=cfg.device  # 🎯 使用全局设备
    )

    # 文本 tokenizer / encoder 可调参数
    cfg.text = SimpleNamespace(
        tokenizer_id="bert-base-uncased",  # 分词器
        encoder_id="bert-base-uncased",  # 文本编码器（可用 sentence-transformers 也行）
        max_length=256,  # 句长上限
        padding="max_length",  # "max_length" / "longest"
        truncation=True,  # 是否截断
        pooling="mean",  # "mean" 或 "cls"
        normalize=True,  # 是否对句向量做 L2 归一化
        device=cfg.device,  # 🎯 使用全局设备
    )

    # -------- Video（仅模型相关可调参数）--------
    cfg.video = SimpleNamespace(
        # 🎯 核心新增: 选择视频backbone
        backbone="videomae",  # 选项: "videomae" | "timesformer" | "slowfast"

        dim=256,  # 输出维度

        # Backbone / 预处理
        model_id="MCG-NJU/videomae-base",  # VideoMAE 等模型 ID
        target_size=(224, 224),  # 输入分辨率（与 backbone 位置编码匹配）
        use_fast_processor=True,  # HF 预处理器（若使用）

        # 时间建模（滑窗）
        clip_len=16,  # 每个 clip 的帧数
        stride=8,  # 相邻 clip 的步长

        # 训练策略（只涉及模型可学习范围）
        freeze_backbone=True,  # True=冻结预训练主干，只训头部
        finetune_last_n_blocks=0,  # >0 时解冻主干最后 N 个 block

        # 🎯 新增: TimeSformer配置
        timesformer_id="facebook/timesformer-base-finetuned-k400",
        timesformer_max_frames=8,  # 每次处理的最大帧数

        # 🎯 新增: SlowFast配置 (可选)
        slowfast_alpha=8,
        slowfast_beta=0.125,
        slowfast_clip_len=32,

        # 采样配置
        frame_stride=1,  # 帧采样步长
        max_frames=None,  # 最大帧数限制

        # MicroLocal分支（可学习头）
        micro=SimpleNamespace(
            enabled=True,  # 是否启用局部表征分支
            d=256,  # 内部通道维度
            tau=0.25,  # 注意力温度
            dropout=0.0,  # Dropout 比例
        ),

        # 融合头（视频全局 + 局部分支）
        fusion=SimpleNamespace(
            n_layers=2,  # TransformerEncoder 层数
            n_heads=4,  # Multi-Head 注意力头数
            dropout=0.1,  # 注意力/FFN dropout
            dim=256,  # 融合输出维度（通常与 video.dim 一致）
        ),

        # 输出归一化/映射
        norm=SimpleNamespace(
            use_ln=True,  # 输出前是否 LayerNorm
            proj_dim=256,  # 若与 fusion.dim 不同，则线性映射到该维
        ),

        # 运行期（仅影响警告/稳定性，不改变训练超参）
        disable_flash=False,  # 关闭 flash SDPA 的尝试/告警（Windows 常用）
    )

    # -------- Corrector（仅模型相关可调参数）--------
    cfg.corrector = SimpleNamespace(
        num_experts=3,  # For MoE model
        hidden_adjuster=4,  # Dimensional scaling factor
    )

    cfg.train = SimpleNamespace(
        batch_size=4
    )

    cfg.fusion = SimpleNamespace(
        num_layer=1,
        hidden_dim=768
    )

    return cfg