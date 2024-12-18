import numpy as np



# IMPULSE NOISE
def impulse_noise(signal: np.array, 
                  *, 
                  low_threshold: float, 
                  high_threshold: float, 
                  size: float = 0.03) -> np.array:
    
    noise_sample = np.random.default_rng().uniform(low_threshold * min(signal), 
                                                   high_threshold * max(signal), 
                                                   int(size * len(signal)))
    zeros = np.zeros(len(signal) - len(noise_sample))
    noise = np.concatenate([noise_sample, zeros])
    np.random.shuffle(noise)
    noised_signal = signal + noise

    return noised_signal



# FREQUENCY NOISE
def frequency_noise(signal: np.array, 
                    samplerate: int,
                    *,
                    min_freq: int, 
                    max_freq: int, 
                    c: float = 0.03) -> np.array:

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
    idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    f[idx] = 1

    return signal + c * fftnoise(f)



# CLIPPING NOISE
def clipping_noise(signal: np.array,
                   *,
                   threshold: float) -> np.array:
    
    boundary = threshold * max(signal)
    noised_signal = signal.copy()
    noised_signal[signal > boundary] = boundary
    noised_signal[signal < -boundary] = -boundary

    return noised_signal



# COLOR NOISES
def noise_psd(N, psd = lambda f: 1):
        X_white = np.fft.rfft(np.random.randn(N));
        S = psd(np.fft.rfftfreq(N))
        # Normalize S
        S = S / np.sqrt(np.mean(S**2))
        X_shaped = X_white * S;
        return np.fft.irfft(X_shaped);

def PSDGenerator(f):
    return lambda N: noise_psd(N, f)

@PSDGenerator
def white_noise(f):
    return 1;

@PSDGenerator
def blue_noise(f):
    return np.sqrt(f);

@PSDGenerator
def violet_noise(f):
    return f;

@PSDGenerator
def brownian_noise(f):
    return 1/np.where(f == 0, float('inf'), f)

@PSDGenerator
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))


def color_noise(signal: np.array,
                    *,
                    color: function,
                    c: float) -> np.array:
    return signal + c * color(signal.shape[0]) 