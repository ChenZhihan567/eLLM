"""AudioWhisperEncoder
    -------------------
    A wrapper for the OpenAI Whisper encoder that extracts audio representations
    and optionally aligns them to a fixed temporal frequency (default: 25 Hz).

    Functionality:
        - Converts raw waveforms into Whisper-compatible log-mel spectrograms
        - Encodes features using Whisper's transformer encoder
        - Optionally aligns temporal resolution to a target rate (e.g., 25 Hz)
        - Returns both the aligned features and a validity mask

    Input:  List of 1D numpy arrays (raw waveforms)
    Output: 
        - hidden_states: (B, T, D) feature tensor
        - mask: (B, T_aligned) boolean mask marking valid frames
"""

import torch
import numpy as np
from transformers import WhisperModel, WhisperFeatureExtractor
from acma.config import get_default_config
from acma.encoders.audio_loader import AudioLoader
import torch.nn.functional as F
from acma.preprocessing.data_loader import dataloader


CFG = get_default_config()

FEATURIZER = WhisperFeatureExtractor.from_pretrained(
            CFG.audio.whisper_id
        )
MODEL = WhisperModel.from_pretrained(CFG.audio.whisper_id)



class AudioWhisperEncoder:
    """Initialize the Whisper encoder and feature extractor.

        Args:
            cfg: Configuration object that defines parameters such as
                 device, sampling rate, Whisper model ID, and encoder freezing.
    """

    def __init__(self, file_list, cfg=CFG):
        self.cfg = cfg

        # Select compute device (prefer GPU if available)
        self.device = torch.device(
            self.cfg.audio.device if torch.cuda.is_available() else "cpu"
        )

        # Initialize Whisper feature extractor and model
        #   - WhisperFeatureExtractor converts waveforms to log-mel spectrograms
        #   - WhisperModel provides the full Transformer (we use only its encoder)
        self.featurizer = FEATURIZER
        
        self.model = MODEL

        self.audio_load(file_list)
        self.target_frames = np.floor(np.array(self.duration) * self.cfg.hard_alignment.target_hz).astype(int)

        # Freeze encoder parameters to use Whisper as a fixed feature extractor
        if self.cfg.audio.freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        # Move model to the selected device
        self.model.to(self.device)

    def extract_features(self, align=True):
        """
        Extract Whisper encoder features from a batch of audio samples.

        Args:
            audio_np (List[np.ndarray]): A list of 1D numpy arrays (raw audio waveforms)
            align (bool): Whether to temporally align the features to a fixed frame rate

        Returns:
            hidden_states (torch.Tensor): Encoder features of shape (B, T, D)
            mask (torch.BoolTensor): Frame validity mask of shape (B, T_aligned)
        """

        
        # Convert raw audio to Whisper-compatible log-mel spectrogram features.
        # NOTE:
        #   Whisper requires all samples in a batch to have the same temporal length (T).
        #   If input waveforms vary in duration, set `padding=max_length` to automatically
        #   pad shorter ones with zeros (silence) so that they can form a valid batch.
        inputs = self.featurizer(
            self.audio_np,
            sampling_rate=self.cfg.audio.target_sr,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_attention_mask=True
        )  # spectrogram

        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Forward through Whisper’s encoder to obtain hidden representations.
        # Shape: (B, T, D)
        with torch.no_grad():
            hidden_states = self.model.get_encoder()(input_features, attention_mask=attention_mask).last_hidden_state  # (B, T, D)

        # print(hidden_states.shape)

        # Compute the number of raw samples per clip (for duration calculation)
        num_samples = np.array([len(sublist) for sublist in self.audio_np])
        num_samples = num_samples.astype(np.float32)
        # print(num_samples.dtype)
        
        mask = None

        # Optionally perform temporal alignment (e.g., 25 Hz)
        if align:
            hidden_states, mask = self.align_features(hidden_states, num_samples)

        return hidden_states, mask

    def align_features(self, hidden_states, num_samples):
        """
        Align Whisper features along the temporal axis to a fixed frequency (default 25 Hz).

        Args:
            hidden_states (torch.Tensor): Whisper output of shape (B, T, D)
            num_samples (np.ndarray): Original waveform lengths in samples

        Returns:
            aligned (torch.Tensor): Aligned feature tensor of shape (B, T_aligned, D)
            mask (torch.BoolTensor): Boolean mask indicating valid frames (B, T_max)
        """

        B, T, D = hidden_states.shape

        max_tar_frames = max(self.target_frames)
        # print("max:",max_tar_frames)
        aligned_list = []

        # Linearly interpolate each sample to its respective frame length
        for i, tf in enumerate(self.target_frames):
            aligned_i = self.linear_interpolation(hidden_states[i: i+1], tf)
            aligned_list.append(aligned_i)

        # Pad all samples to the same length (B, max_T, D)
        aligned = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in aligned_list], batch_first=True)

        # Construct a boolean mask marking valid frames (True = valid, False = padded)
        mask = torch.zeros((B, max_tar_frames), dtype=torch.bool, device=self.cfg.audio.device)
        for b in range(B):
            mask[b, :self.target_frames[b]] = True
        
        return aligned, mask


    def linear_interpolation(self, hidden_states, target_frame):
        """
        Perform 1D linear interpolation along the temporal dimension.

        Args:
            hidden_states (torch.Tensor): Input features (B, T, D)
            target_frame (int): Target number of frames after resampling

        Returns:
            aligned (torch.Tensor): Resampled features of shape (B, target_frame, D)
        """

        # Rearrange to (B, D, T) for torch.interpolate compatibility
        x = hidden_states.permute(0, 2, 1)

        # Perform linear interpolation (align_corners=False ensures proportional scaling)
        aligned = F.interpolate(x, size=target_frame, mode="linear", align_corners=False)
        
        # Restore original order: (B, T, D)
        aligned = aligned.permute(0, 2, 1) 
        return aligned


    def audio_load(self, file_list):
        al = AudioLoader(file_list, self.cfg)
        self.audio_np, self.duration = al.load_all_audios()
        return 




if __name__ == "__main__":
    

    dl = dataloader()

    for idx, batch in enumerate(dl):
        awe = AudioWhisperEncoder(batch)
        encoding, mask = awe.extract_features()
        print(f"Audio encoding {idx}: ", encoding.shape)


    

    