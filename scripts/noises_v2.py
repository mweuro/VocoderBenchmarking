import torch
import numpy as np
from typing import Union, Literal



def _array_or_tensor(signal: np.array, 
                     output_format: Literal['numpy', 'tensor'] = 'numpy') -> Union[np.ndarray, torch.Tensor]:
    if output_format == 'numpy':
        pass
    elif output_format == 'tensor':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        signal = torch.Tensor(signal).to(device)
    else:
        raise ValueError("Output format must be either 'numpy' or 'tensor'")

    return signal
        


def _noise_psd(N, psd = lambda f: 1) -> np.ndarray:
    X_white = np.fft.rfft(np.random.randn(N))
    S = psd(np.fft.rfftfreq(N))
    S = S / np.sqrt(np.mean(S**2))
    X_shaped = X_white * S
    noise = np.fft.irfft(X_shaped)
    
    return np.append(noise, 0.) if N % 2 == 1 else noise






class Noise: pass



class WhiteNoise(Noise):
    def __init__(self, c: float) -> None:
        self.c = c
    
    def __call__(self, 
                 signal: Union[np.ndarray, torch.Tensor],
                 *,
                 output_format: Literal['numpy', 'tensor'] = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        
        if type(signal) == torch.Tensor:
            signal = signal.cpu().numpy()
        
        color_noise = _noise_psd(len(signal), lambda f: 1)
        noised_signal = signal + self.c * color_noise
        output_signal = _array_or_tensor(noised_signal, output_format = output_format)
        
        return output_signal



class BlueNoise(Noise):
    def __init__(self, c: float) -> None:
        self.c = c
    
    def __call__(self, 
                 signal: Union[np.ndarray, torch.Tensor],
                 *,
                 output_format: Literal['numpy', 'tensor'] = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        
        if type(signal) == torch.Tensor:
            signal = signal.cpu().numpy()
        
        color_noise = _noise_psd(len(signal), lambda f: np.sqrt(f))
        noised_signal = signal + self.c * color_noise
        output_signal = _array_or_tensor(noised_signal, output_format = output_format)
        
        return output_signal



class VioletNoise(Noise):
    def __init__(self, c: float) -> None:
        self.c = c
    
    def __call__(self, 
                 signal: Union[np.ndarray, torch.Tensor],
                 *,
                 output_format: Literal['numpy', 'tensor'] = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        
        if type(signal) == torch.Tensor:
            signal = signal.cpu().numpy()
        
        color_noise = _noise_psd(len(signal), lambda f: f)
        noised_signal = signal + self.c * color_noise
        output_signal = _array_or_tensor(noised_signal, output_format = output_format)
        
        return output_signal



class BrownianNoise(Noise):
    def __init__(self, c: float) -> None:
        self.c = c
    
    def __call__(self, 
                 signal: Union[np.ndarray, torch.Tensor],
                 *,
                 output_format: Literal['numpy', 'tensor'] = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        
        if type(signal) == torch.Tensor:
            signal = signal.cpu().numpy()
        
        color_noise = _noise_psd(len(signal), lambda f: 1 / np.where(f == 0, float('inf'), f))
        noised_signal = signal + self.c * color_noise
        output_signal = _array_or_tensor(noised_signal, output_format = output_format)
        
        return output_signal



class PinkNoise(Noise):
    def __init__(self, c: float) -> None:
        self.c = c
    
    def __call__(self, 
                 signal: Union[np.ndarray, torch.Tensor],
                 *,
                 output_format: Literal['numpy', 'tensor'] = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        
        if type(signal) == torch.Tensor:
            signal = signal.cpu().numpy()
        
        color_noise = _noise_psd(len(signal), lambda f: 1 / np.where(f == 0, float('inf'), np.sqrt(f)))
        noised_signal = signal + self.c * color_noise
        output_signal = _array_or_tensor(noised_signal, output_format = output_format)
        
        return output_signal
    
        

class ImpulseNoise(Noise):
    def __init__(self,
                 *,
                 low_threshold: float, 
                 high_threshold: float, 
                 size: float = 0.0005) -> None:
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.size = size

    def __call__(self, 
                 signal: Union[np.ndarray, torch.Tensor],
                 *,
                 output_format: Literal['numpy', 'tensor'] = 'numpy') -> Union[np.ndarray, torch.Tensor]:

        if type(signal) == torch.Tensor:
            signal = signal.cpu().numpy()
            
        noise_sample = np.random.default_rng().\
            uniform(self.low_threshold * min(signal), 
                    self.high_threshold * max(signal), 
                    int(self.size * len(signal)))   
        zeros = np.zeros(len(signal) - len(noise_sample))
        noise = np.concatenate([noise_sample, zeros])
        np.random.shuffle(noise)
        noised_signal = signal + noise
        output_signal = _array_or_tensor(noised_signal, output_format = output_format)
        
        return output_signal



class FrequencyNoise(Noise):
    def __init__(self,
                 *,
                 min_freq: int, 
                 max_freq: int, 
                 c: float = 0.03) -> None:
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.c = c

    def __call__(self, 
                 signal: Union[np.ndarray, torch.Tensor],
                 samplerate: int = 22050,
                 *,
                 output_format: Literal['numpy', 'tensor'] = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        
        if type(signal) == torch.Tensor:
            signal = signal.cpu().numpy()
            
        def fftnoise(f: np.array) -> np.array:
            f = np.array(f, dtype = 'complex')
            Np = (len(f) - 1) // 2
            phases = np.random.rand(Np) * 2 * np.pi
            phases = np.cos(phases) + 1j * np.sin(phases)
            f[1 : (Np + 1)] *= phases
            f[-1 : (-1 - Np) : -1] = np.conj(f[1 : (Np + 1)])
            return np.fft.ifft(f).real
            
        samples = len(signal)
        freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
        f = np.zeros(samples)
        idx = np.where(np.logical_and(freqs >= self.min_freq, freqs <= self.max_freq))[0]
        f[idx] = 1
        noised_signal = signal + self.c * fftnoise(f)
        output_signal = _array_or_tensor(noised_signal, output_format = output_format)
        
        return output_signal



class ClippingNoise(Noise):
    def __init__(self,
                 *,
                 threshold: float) -> None:
        self.threshold = threshold

    def __call__(self, 
                 signal: Union[np.ndarray, torch.Tensor],
                 *,
                 output_format: Literal['numpy', 'tensor'] = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        
        if type(signal) == torch.Tensor:
            signal = signal.cpu().numpy()
        
        boundary = self.threshold * max(signal)
        noised_signal = signal.copy()
        noised_signal[signal > boundary] = boundary
        noised_signal[signal < -boundary] = -boundary
        output_signal = _array_or_tensor(noised_signal, output_format = output_format)
        
        return output_signal