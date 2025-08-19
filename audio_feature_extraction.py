import torch
import torchaudio
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, NUM_SAMPLES

class AudioFeatureExtractor:
    def __init__(self):
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        self.resampler = None

    def _resample_if_needed(self, waveform, sr):
        if sr != SAMPLE_RATE:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            return self.resampler(waveform)
        return waveform

    def _mix_down_if_needed(self, waveform):
        if waveform.shape[0] > 1:  # More than 1 channel
            return torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _pad_or_truncate(self, waveform):
        if waveform.shape[1] > NUM_SAMPLES:
            return waveform[:, :NUM_SAMPLES]
        if waveform.shape[1] < NUM_SAMPLES:
            padding = NUM_SAMPLES - waveform.shape[1]
            return torch.nn.functional.pad(waveform, (0, padding))
        return waveform

    def transform(self, waveform, sr):
        waveform = self._resample_if_needed(waveform, sr)
        waveform = self._mix_down_if_needed(waveform)
        waveform = self._pad_or_truncate(waveform)
        mel_spec = self.mel_spectrogram(waveform)
        return mel_spec
