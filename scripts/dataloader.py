import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AudioDataset(Dataset):

    def __init__(self, root, sample_rate = 22050):
        self.root = root
        self.sample_rate = sample_rate
        self.audio_paths = self._get_audio_paths()
        self.max_length = self._find_max_length()

    def _get_audio_paths(self):
        return list(Path(self.root).rglob('*.wav'))

    def _find_max_length(self):
        lengths = [sf.info(path).frames for path in self.audio_paths]
        return max(lengths)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        y, _ = librosa.load(audio_path, sr = self.sample_rate)
        padded_y = np.pad(y, (0, max(0, self.max_length - len(y))), mode = 'constant')
        y = torch.tensor(padded_y, dtype=torch.float32).to(DEVICE)
        
        return y, self.sample_rate



def create_dataloader(dataset, batch_size, shuffle, num_workers):

    return DataLoader(dataset, 
                      batch_size = batch_size, 
                      shuffle = shuffle, 
                      num_workers = num_workers)