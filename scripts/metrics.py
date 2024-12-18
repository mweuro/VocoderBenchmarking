import numpy as np
import librosa
import torch
import torchaudio
from scipy.linalg import sqrtm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from pymcd.mcd import Calculate_MCD
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from tqdm import tqdm
import random


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')


def load_audio(file_path):
    """Wczytuje plik WAV i resampluje do target_sr."""
    waveform, sr = torchaudio.load(file_path)
    waveform = waveform.mean(dim=0)  # Konwersja do mono
    return waveform


def calculate_single_sdr(src_wav, gen_wav):
    src_spec = librosa.stft(src_wav.cpu().numpy())
    gen_spec = librosa.stft(gen_wav.cpu().numpy())

    src_log = librosa.amplitude_to_db(np.abs(src_spec), ref=np.max)
    gen_log = librosa.amplitude_to_db(np.abs(gen_spec), ref=np.max)

    min_len = min(src_log.shape[1], gen_log.shape[1])
    src_log = src_log[:, :min_len]
    gen_log = gen_log[:, :min_len]

    noise = src_log - gen_log
    sdr = 10 * np.log10(np.sum(np.abs(src_log) ** 2) / np.sum(np.abs(noise) ** 2))
    return sdr


def calculate_sdr(src_wavs, gen_wavs):
    """
    all wavs should be in the same sample rate
    """
    sdrs = []
    for src_wav, gen_wav in zip(src_wavs, gen_wavs):
        sdrs.append(calculate_single_sdr(src_wav, gen_wav))
    return np.mean(sdrs)


def extract_embeddings(processor, model, waveforms):
    """Ekstrakcja embeddingów za pomocą pretrenowanego modelu Wav2Vec 2.0."""
    resampler = torchaudio.transforms.Resample(48000, 16000)
    inputs = processor(pad_sequence([resampler(wav) for wav in waveforms], batch_first=True), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(input_values=inputs['input_values'].squeeze(0))
    # Średnia po czasie dla globalnych embeddingów
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()


def calculate_fad(embeddings1, embeddings2):
    """Oblicza Frechet Audio Distance (FAD) między dwoma zbiorami embeddingów."""
    embeddings1 = np.atleast_2d(embeddings1)
    embeddings2 = np.atleast_2d(embeddings2)

    mean1, mean2 = np.mean(embeddings1, axis=0), np.mean(embeddings2, axis=0)
    cov1, cov2 = np.cov(embeddings1, rowvar=False), np.cov(embeddings2, rowvar=False)
    
    if cov1.ndim == 0 or cov2.ndim == 0:  # Jeśli covariancje są skalarami
        cov1 = np.array([[cov1]])
        cov2 = np.array([[cov2]])
    
    cov_sqrt = sqrtm(cov1 @ cov2)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    fad = np.sum((mean1 - mean2) ** 2) + np.trace(cov1 + cov2 - 2 * cov_sqrt)
    return fad


def calculate_kid(embeddings1, embeddings2, kernel='rbf', gamma=None):
    """Oblicza Kernel Inception Distance (KID) między dwoma zbiorami embeddingów."""
    
    def kernel_matrix(X, Y=None):
        if kernel == 'rbf':
            return rbf_kernel(X, Y, gamma=gamma)
        elif kernel == 'polynomial':
            return polynomial_kernel(X, Y, degree=3, coef0=1)
        else:
            raise ValueError(f"Nieobsługiwany kernel: {kernel}")
    
    K_XX = kernel_matrix(embeddings1)
    K_YY = kernel_matrix(embeddings2)
    K_XY = kernel_matrix(embeddings1, embeddings2)
    
    m, n = embeddings1.shape[0], embeddings2.shape[0]
    kid = (np.sum(K_XX) / (m * m) + np.sum(K_YY) / (n * n) - 2 * np.sum(K_XY) / (m * n))
    return kid


def calculate_mcd(src_files, gen_files):
    results = []
    for src, gen in zip(src_files, gen_files):
        results.append(mcd_toolbox.calculate_mcd(src, gen))
    return np.mean(results)


def calculate_pesq(src_wavs, gen_wavs):
    results = []
    for src, gen in zip(src_wavs, gen_wavs):
        min_len = min(len(src), len(gen))
        src = src[:min_len]
        gen = gen[:min_len]
        results.append(pesq(src, gen))
    return np.mean(results)


def calculate_metrics(src_wavs, gen_wavs, src_embeddings=None, gen_embeddings=None):
    '''
    Jeżeli zamiast ścieżki do wczytania dostarczane są wcześniej wczytane wav, powinny mieć sr=48000
    '''
    results = {}

    src_wavs_str, gen_wavs_str = [], []
    for src, gen in zip(src_wavs, gen_wavs):
        if isinstance(src, str) and isinstance(gen, str):
            src_wavs_str.append(src)
            gen_wavs_str.append(gen)
    if src_wavs_str:
        results['mcd'] = calculate_mcd(src_wavs_str, gen_wavs_str)

    src_wavs = [load_audio(wav) if isinstance(wav, str) else wav.torch() if isinstance(wav, np.ndarray) else wav for wav in src_wavs]
    gen_wavs = [load_audio(wav) if isinstance(wav, str) else wav.torch() if isinstance(wav, np.ndarray) else wav for wav in gen_wavs]

    if src_embeddings is None:
        src_embeddings = extract_embeddings(processor, model, pad_sequence(src_wavs, batch_first=True))
    if gen_embeddings is None:
        gen_embeddings = extract_embeddings(processor, model, pad_sequence(gen_wavs, batch_first=True))

    results['pesq'] = calculate_pesq(src_wavs, gen_wavs)
    results['sdr'] = calculate_sdr(src_wavs, gen_wavs)
    results['fad'] = calculate_fad(src_embeddings, gen_embeddings)
    results['kid'] = calculate_kid(src_embeddings, gen_embeddings)

    return results


def calculate_metrics_for_all_data(src_paths, gen_paths_list, model_names, n_split=10):
    '''
    funkcja zoptymalizowana pod wyliczanie wszystkich metryk na raz dla każdego modelu
    '''
    results =  {model_name: {'sdr': [], 'fad': [], 'kid': [], 'mcd': [], 'pesq': []} for model_name in model_names}

    random.seed(2137)
    indexes = list(range(len(src_paths)))[:]
    random.shuffle(indexes)
    section_range = len(indexes) // n_split
    rest = len(indexes) % n_split
    sections = []
    start = 0
    for i in range(n_split):
        dodatkowy = 1 if i < rest else 0
        end = start + section_range + dodatkowy
        sections.append(set(indexes[start:end]))
        start = end

    for section in tqdm(sections):
        section_src_paths = [path for i, path in enumerate(src_paths) if i in section]
        src_wavs = [load_audio(path) for path in section_src_paths]
        src_log_specs = [librosa.amplitude_to_db(np.abs(librosa.stft(wav.cpu().numpy())), ref=np.max) for wav in src_wavs]
        src_embeddings = extract_embeddings(processor, model, src_wavs)
        src_mean = np.mean(src_embeddings, axis=0)
        src_cov = np.cov(src_embeddings, rowvar=False)
        src_K = rbf_kernel(src_embeddings)
        # src_wavs_int = [(wav.numpy() * 32767).astype(np.int16) for wav in src_wavs]

        for gen_paths, model_name in tqdm(zip(gen_paths_list, model_names)):
            result = results[model_name]
            section_gen_paths = [path for i, path in enumerate(gen_paths) if i in section]
            gen_wavs = [load_audio(path) for path in section_gen_paths]
            gen_log_specs = [librosa.amplitude_to_db(np.abs(librosa.stft(wav.cpu().numpy())), ref=np.max) for wav in gen_wavs]
            gen_embeddings = extract_embeddings(processor, model, gen_wavs)
            gen_mean = np.mean(gen_embeddings, axis=0)
            gen_cov = np.cov(gen_embeddings, rowvar=False)
            gen_K = rbf_kernel(gen_embeddings)
            # gen_wavs_int = [(wav.numpy() * 32767).astype(np.int16) for wav in gen_wavs]

            # SDR
            sdrs = []
            for src, gen in zip(src_log_specs, gen_log_specs):
                min_len = min(src.shape[1], gen.shape[1])
                src = src[:, :min_len]
                gen = gen[:, :min_len]
                noise = src - gen
                sdr = 10 * np.log10(np.sum(np.abs(src) ** 2) / np.sum(np.abs(noise) ** 2))
                sdrs.append(sdr)
            result['sdr'].append(np.mean(sdrs))

            # FAD
            cov_sqrt = sqrtm(src_cov @ gen_cov)
            if np.iscomplexobj(cov_sqrt):
                cov_sqrt = cov_sqrt.real
            fad = np.sum((src_mean - gen_mean) ** 2) + np.trace(src_cov + gen_cov - 2 * cov_sqrt)
            result['fad'].append(fad)

            # KID
            src_gen_K = rbf_kernel(src_embeddings, gen_embeddings)
            m, n = src_embeddings.shape[0], gen_embeddings.shape[0]
            kid = (np.sum(src_K) / (m * m) + np.sum(gen_K) / (n * n) - 2 * np.sum(src_gen_K) / (m * n))
            result['kid'].append(kid)

            # MCD
            mcds = []
            for src, gen in zip(section_src_paths, section_gen_paths):
                mcds.append(mcd_toolbox.calculate_mcd(src, gen))
            result['mcd'].append(np.mean(mcds))

            # pesq
            pesqs = []
            for src, gen in zip(src_wavs, gen_wavs):
                min_len = min(len(src), len(gen))
                src = src[:min_len]
                gen = gen[:min_len]
                try:
                    pesqs.append(pesq(src, gen))
                except:
                    pesqs.append(0)
            result['pesq'].append(np.mean(pesqs))

    for model_name in model_names:
        result = results[model_name]
        result['sdr'] = np.mean(result['sdr'])
        result['fad'] = np.mean(result['fad'])
        result['kid'] = np.mean(result['kid'])
        result['mcd'] = np.mean(result['mcd'])
        result['pesq'] = np.mean(result['pesq'])
    
    return results