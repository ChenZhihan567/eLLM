from torch.utils.data import Dataset, DataLoader
from acma.config import get_default_config
# from acma.encoders.audio_whisper import AudioWhisperEncoder 
# from acma.encoders.text_encoder import TextEncoder
import subprocess, shutil
import os


cfg = get_default_config()


class DataGenerator(Dataset):
    """Training Dataset for loading and preprocessing video files.

    This dataset:
      - Loads video files from the directory specified in `config: cfg.dataloader.path`.
      - Ensures all files are converted into a normalized `.mp4` format
        (using ffmpeg with H.264 codec and AAC audio).
      - Stores the processed file paths for downstream.

    Args:
        Dataset (torch.utils.data.Dataset): Inherits from PyTorch's Dataset base class.

    Attributes:
        data_path (str): Directory where raw video files are located.
        data (List[str]): List of processed video file paths in normalized format.

    
    """
    def __init__(self):
        self.data_path = cfg.dataloader.path
        self.load()
        

    def load(self):

        self.data = self.type_check()

        return
    

    def type_check(self):
        """Check and normalize video file formats in the data path.

        Iterates over all files in `self.data_path`. If a file is not in `.mp4` format,
        it will be passed to `self.video_formator()` for conversion to a standardized `.mp4`.
        All resulting filenames (original `.mp4` or converted) are collected and returned.

        Returns:
            List[str]: A list of video filenames in standardized `.mp4` format.
        """
        format_path = []
        for f in os.listdir(self.data_path):
            if not f.endswith(".mp4"):
                #print(f)
                f = self.video_formator(f)
                pass

            format_path.append(f)

        return format_path
    

    def video_formator(self, input_path):
        """ Convert a video file to standardized .mp4 format using ffmpeg.

        This function takes a video file (e.g., .mov, .avi, etc.) from the dataset path
        and re-encodes it into a normalized .mp4 format using H.264 for video and AAC for audio.
        The converted file will be saved in the same directory with a `_norm.mp4` suffix.

        Args:
            input_path (str): The name of the input video file (relative to self.data_path).

        Raises:
            RuntimeError: If ffmpeg is not installed or available in the system PATH.

        Returns:
            str: The filename of the newly generated normalized .mp4 video (e.g., 'xxx_norm.mp4').
        """


        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("The system could not find ffmpeg, please install ffmpeg first: sudo apt install ffmpeg OR pixi add ffmpeg")

        new_f = os.path.splitext(os.path.basename(input_path))[0] + "_norm.mp4"
        ouput_path = os.path.join(self.data_path, new_f)

        input_abs_path = os.path.join(self.data_path, input_path)

        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_abs_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            ouput_path
        ]
        
        subprocess.run(cmd, check=True)

        return new_f


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    


def dataloader(shuffle=False):
    """Create a PyTorch DataLoader for the training dataset.

    Args:
        shuffle (bool, optional): Whether to shuffle the dataset at the beginning of each epoch.
            Defaults to True.

    Returns:
        dl (DataLoader): A PyTorch DataLoader instance that provides mini-batches of training data.
    """
    dg = DataGenerator()
    dl = DataLoader(dg, batch_size=cfg.train.batch_size, shuffle=shuffle)
    return dl




if __name__ == "__main__":
    # dl = dataloader()
    # for batch in dl:
    #     print(batch)
    #     audio_encoder = AudioWhisperEncoder(cfg)
    #     audio_encoder.extract_features()

    #     # text_encoder = TextEncoder(cfg)
    #     # text_encoder.batch_extract(batch)
    #     break

    dl = dataloader()
    for batch in dl:
        print(batch)




