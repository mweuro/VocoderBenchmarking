import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path, PosixPath
import random
from .noises_v2 import Noise





class AudioDataset(Dataset):
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, root: str, sample_rate: int = 22050, transforms: Noise = None) -> None:
        self.root = root
        self.sample_rate = sample_rate
        self.transforms = transforms
        self.audio_paths = self._get_audio_paths()
        self.max_length = self._find_max_length()

    def _get_audio_paths(self) -> list[PosixPath]:
        return list(Path(self.root).rglob('*.wav'))

    def _find_max_length(self) -> int:
        wav_infos = [sf.info(path) for path in self.audio_paths]
        lengths = np.array([meta.frames for meta in wav_infos]) / np.array([meta.samplerate for meta in wav_infos]) * self.sample_rate
        return max(list(np.ceil(lengths).astype('int')))

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        audio_path = self.audio_paths[idx]
        y, _ = librosa.load(audio_path, sr = self.sample_rate)
        padded_y = np.pad(y, (0, max(0, self.max_length - len(y))), mode='constant')
        y = torch.tensor(padded_y, dtype = torch.float32).to(AudioDataset.DEVICE)
        if self.transforms:
            y = self.transforms(y).to(AudioDataset.DEVICE)

        return y, self.sample_rate



class AudioTransforms:
    def __init__(self, transforms: list[Noise]) -> torch.Tensor:
        self.transforms = transforms

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        transform = random.choice(self.transforms + [None])
        if transform:
            signal = torch.tensor(transform(signal), dtype = torch.float32)
            
        return signal



def create_dataloader(dataset: AudioDataset, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, 
                      batch_size = batch_size, 
                      shuffle = shuffle, 
                      num_workers = num_workers)