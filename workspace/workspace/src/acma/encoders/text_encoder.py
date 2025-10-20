import torch
import torch.nn as nn
from acma.config import get_default_config
from acma.encoders.audio_loader import AudioLoader
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    AutoModel,
)

CFG = get_default_config()


class TextEncoder:
    """
    TextEncoder 负责三步：
    1) ASR 转录（Whisper）
    2) Tokenizer 分词
    3) 文本编码为句级 embedding
    """

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else get_default_config()

        # ====== ASR（Whisper）======
        self.asr_device = torch.device(self.cfg.audio.device if torch.cuda.is_available() else "cpu")
        self.asr_processor = WhisperProcessor.from_pretrained(self.cfg.asr.whisper_id)
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(self.cfg.asr.whisper_id)
        self.asr_model.to(self.asr_device)
        self.asr_model.eval()

        # ====== 文本分词器 / 编码器 ======
        self.txt_device = torch.device(self.cfg.text.device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.text.tokenizer_id, use_fast=True)
        self.text_model = AutoModel.from_pretrained(self.cfg.text.encoder_id)
        self.text_model.to(self.txt_device)
        self.text_model.eval()
        in_features = self.text_model.config.hidden_size
        self.fc1 = nn.Linear(in_features, in_features * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features * 4, 2)

        self.fc1.to("cuda")
        self.fc2.to("cuda")


    def transcribe(self, audio_np):

        inputs = self.asr_processor(
            audio_np,
            sampling_rate=self.cfg.audio.target_sr,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            return_attention_mask=True
        ).to(self.asr_device)

        input_features = inputs.input_features
        attention_mask = inputs.attention_mask 

        with torch.no_grad():
            pred_ids = self.asr_model.generate(
                        input_features=input_features,
                        attention_mask=attention_mask,
                        task="transcribe",
                        language="en"
            )
        
        text = self.asr_processor.batch_decode(pred_ids, skip_special_tokens=True)
        return text
    

    def tokenize(self, text: str):
        """
        使用预训练分词器将文本转为 input_ids / attention_mask
        返回字典张量，已放到 self.txt_device 上。
        """
        enc = self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        )
        return {k: v.to(self.txt_device) for k, v in enc.items()}
    

    def extract_features(self, audio_np):

        text = self.transcribe(audio_np)

        sentence_tokens = self.tokenize(text)
        # print(sentence_tokens["input_ids"].shape)

        with torch.no_grad():
            outputs = self.text_model(
                input_ids = sentence_tokens["input_ids"],
                attention_mask = sentence_tokens["attention_mask"],
                token_type_ids = sentence_tokens["token_type_ids"]
            )

        last_hidden_state = outputs.last_hidden_state 
        attention_mask = sentence_tokens["attention_mask"]


        # print(sentence_tokens["attention_mask"].shape)
        # print(sentence_tokens["attention_mask"])


        sentence_feature = self.masked_mean_pooling(last_hidden_state, attention_mask)

        VA_text = torch.tanh(self.fc2(self.act(self.fc1(sentence_feature))))

        return sentence_feature, VA_text
    

    def masked_mean_pooling(self, last_hidden_state, attention_mask):
   
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state) 
        summed = (last_hidden_state * mask).sum(dim=1)                  
        count = mask.sum(dim=1).clamp(min=1e-9)                        
        return summed / count                                          



      

if __name__ == "__main__":
 
    al = AudioLoader(CFG, ['dia2_utt1.mp4', 'dia1_utt15.mp4', 'dia1_utt0.mp4', 'dia1_utt4.mp4'])
    # al = AudioLoader(CFG, ['dia2_utt1.mp4'])

    wave = al.load_all_audios()
    te = TextEncoder(CFG)
    encoding = te.extract_features(wave)
    print(encoding.shape)