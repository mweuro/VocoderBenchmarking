import numpy as np
import torch
import librosa
from .noises_v2 import *
import matplotlib
import matplotlib.pyplot as plt
from typing import Union



def _prepare_wave(signal: Union[np.ndarray, torch.Tensor], 
                  sr: int = 22050, 
                  title: str = 'Signal Wave', 
                  ax: matplotlib.axes.Axes = None) -> None:
    
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()
    time_axis = (torch.arange(0, len(signal)) / sr)

    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.plot(time_axis, signal, linewidth = 1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy')



def _prepare_spectogram(signal: Union[np.ndarray, torch.Tensor], 
                    sr: int = 22050, 
                    n_fft: int = 2048,
                    hop_length: int = 64,
                    title: str = 'Signal Spectogram (dB)',
                    ax: matplotlib.axes.Axes = None) -> None:
    
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()
        
    S = librosa.stft(signal, n_fft = n_fft, hop_length = hop_length)
    D = librosa.amplitude_to_db(abs(S), ref=np.max)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()
        
    spec = librosa.display.specshow(D, sr = sr, hop_length = hop_length, 
                             x_axis = 'time', 
                             y_axis = 'log', 
                             cmap = 'viridis',
                             ax = ax)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    
    return spec



def plot_wave(signal: Union[np.ndarray, torch.Tensor], 
              *,
              sr: int = 22050, 
              title: str = 'Signal Wave',
              figsize: tuple[int] = (15, 5)) -> None:
    
    _, ax = plt.subplots(1, 1, figsize = figsize)
    _prepare_wave(signal, sr, title, ax = ax)
    plt.show()



def plot_spec(signal: Union[np.ndarray, torch.Tensor], 
              *,
              sr: int = 22050, 
              n_fft: int = 2048,
              hop_length: int = 64,
              title: str = 'Signal Spectogram (dB)',
              figsize: tuple[int] = (15, 5)) -> None:
    
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    spec = _prepare_spectogram(signal, sr, n_fft, hop_length, title, ax = ax)
    
    cbar_ax = fig.add_axes([0.2, -0.05, 0.6, 0.1])
    cbar_ax.set_axis_off()
    fig.colorbar(spec, ax = cbar_ax, format = '%+2.0f dB', orientation = "horizontal", fraction = 0.5)
    plt.show()



def plot_wave_and_spec(signal: Union[np.ndarray, torch.Tensor], 
                       *,
                       sr: int = 22050, 
                       n_fft: int = 2048,
                       hop_length: int = 64,
                       title_wave: str = 'Signal Wave',
                       title_spec: str = 'Signal Spectogram (dB)',
                       figsize: tuple[int] = (15, 10)) -> None:
    
    fig, axs = plt.subplots(2, 1, figsize = figsize)
    _prepare_wave(signal, sr, title_wave, ax = axs[0])
    spec = _prepare_spectogram(signal, sr, n_fft, hop_length, title_spec, ax = axs[1])
    cbar_ax = fig.add_axes([0.2, -0.05, 0.6, 0.1])
    cbar_ax.set_axis_off()
    fig.colorbar(spec, ax = cbar_ax, format = '%+2.0f dB', orientation = "horizontal", fraction = 0.5)

    plt.tight_layout()
    plt.show()



def plot_multiple_waves(signals: list[Union[np.ndarray, torch.Tensor]], 
                        *,
                        sr: int = 22050, 
                        titles: list[str] = None,
                        figsize: tuple[int] = None) -> None:
    
    if titles is None:
        titles = len(signals) * [None]
    if len(titles) != len(signals):
        raise ValueError('Incorrect number of titles')
    if figsize is None:
        figsize = (15, 5*len(signals))
    no_signals = len(signals)
    
    _, axs = plt.subplots(no_signals, 1, figsize = figsize)
    for i, (signal, title) in enumerate(zip(signals, titles)):
        _prepare_wave(signal, sr, title, ax = axs[i])
        
    plt.tight_layout()
    plt.show()



def plot_multiple_specs(signals: list[Union[np.ndarray, torch.Tensor]], 
                        *,
                        sr: int = 22050, 
                        n_fft: int = 2048,
                        hop_length: int = 64,
                        titles: list[str] = None,
                        figsize: tuple[int] = None) -> None:
    
    if titles is None:
        titles = len(signals) * [None]
    if len(titles) != len(signals):
        raise ValueError('Incorrect number of titles')
    if figsize is None:
        figsize = (15, 5*len(signals))
    no_signals = len(signals)
    
    fig, axs = plt.subplots(no_signals, 1, figsize = figsize)
    for i, (signal, title) in enumerate(zip(signals, titles)):
        spec = _prepare_spectogram(signal, sr, n_fft, hop_length, title, ax = axs[i])
    cbar_ax = fig.add_axes([0.2, -0.05, 0.6, 0.1])
    cbar_ax.set_axis_off()
    fig.colorbar(spec, ax = cbar_ax, format = '%+2.0f dB', orientation = "horizontal", fraction = 0.5)
     
    
    plt.tight_layout()
    plt.show()