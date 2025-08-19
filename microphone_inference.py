import torch
import sounddevice as sd
import numpy as np
import time
import os
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
import noisereduce as nr
import webrtcvad
from scipy.signal import butter, filtfilt
from pathlib import Path

from config import *
from models import CNNModel, RNNModel
from audio_feature_extraction import AudioFeatureExtractor

class AudioProcessor:
    """
    A class to record and process audio, featuring noise reduction,
    voice activity detection, and normalization.
    """
    def __init__(self, sample_rate: int = MIC_SAMPLE_RATE, duration: float = 2.0, target_duration: float = 1.4):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_duration = target_duration
        self.vad = webrtcvad.Vad(3)
        self.results_dir = Path("live_recordings")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print("Audio Processor Initialized")

    def record_audio(self):
        """Records audio from the default input device with a countdown."""
        print("\nStarting audio recording...")
        for i in range(3, 0, -1):
            print(f"Recording in {i}...")
            time.sleep(1)
        
        print("RECORDING...")
        try:
            recording = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            print("Recording complete.")
            return recording.flatten()
        except Exception as e:
            print(f"Error during recording: {e}")
            return None

    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Executes the complete audio processing pipeline."""
        if audio is None or len(audio) == 0:
            return np.array([], dtype=np.float32)

        filtered_audio = self._apply_high_pass_filter(audio)
        noise_profile = filtered_audio[:int(0.25 * self.sample_rate)]
        denoised_audio = self._reduce_noise_with_profile(filtered_audio, noise_profile)
        start, end = self._trim_silence(denoised_audio)
        trimmed_audio = denoised_audio[start:end]

        if len(trimmed_audio) == 0:
            return np.array([], dtype=np.float32)

        duration_normalized_audio = self._normalize_duration(trimmed_audio)
        final_audio = self._normalize_loudness(duration_normalized_audio)
        
        if np.max(np.abs(final_audio)) > 0:
            final_audio = final_audio / np.max(np.abs(final_audio)) * 0.98
        return final_audio

    def _apply_high_pass_filter(self, audio: np.ndarray, cutoff: int = 100) -> np.ndarray:
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(4, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, audio)

    def _reduce_noise_with_profile(self, audio: np.ndarray, noise_profile: np.ndarray) -> np.ndarray:
        return nr.reduce_noise(y=audio, y_noise=noise_profile, sr=self.sample_rate)

    def _trim_silence(self, audio: np.ndarray) -> tuple[int, int]:
        frame_duration_ms = 20
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)
        audio_int16 = np.int16(audio * 32767)
        
        speech_frames = []
        for i in range(0, len(audio_int16), frame_size):
            frame = audio_int16[i:i+frame_size]
            if len(frame) < frame_size: break
            is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
            speech_frames.append(is_speech)

        if not any(speech_frames): return 0, 0
        
        first_speech_frame = np.where(speech_frames)[0][0]
        last_speech_frame = np.where(speech_frames)[0][-1]
        
        padding_frames = 5
        start_frame = max(0, first_speech_frame - padding_frames)
        end_frame = min(len(speech_frames) - 1, last_speech_frame + padding_frames)
        
        return start_frame * frame_size, (end_frame + 1) * frame_size

    def _normalize_duration(self, audio: np.ndarray) -> np.ndarray:
        target_samples = int(self.target_duration * self.sample_rate)
        if len(audio) > target_samples:
            start = (len(audio) - target_samples) // 2
            return audio[start:start + target_samples]
        padding = target_samples - len(audio)
        return np.pad(audio, (padding // 2, padding - (padding // 2)), mode='constant')
        
    def _normalize_loudness(self, audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        current_rms = np.sqrt(np.mean(audio**2))
        return audio * (target_rms / current_rms) if current_rms > 1e-4 else audio

    def create_comparison_plot(self, original: np.ndarray, enhanced: np.ndarray, save_path: Path):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Audio Processing Comparison', fontsize=16)

        librosa.display.waveshow(original, sr=self.sample_rate, ax=axes[0, 0], color='royalblue')
        axes[0, 0].set_title('Original Waveform')
        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original)), ref=np.max)
        librosa.display.specshow(D_orig, sr=self.sample_rate, x_axis='time', y_axis='hz', ax=axes[1, 0])
        axes[1, 0].set_title('Original Spectrogram')

        if enhanced.size > 0:
            librosa.display.waveshow(enhanced, sr=self.sample_rate, ax=axes[0, 1], color='forestgreen')
            axes[0, 1].set_title('Enhanced Waveform')
            D_enh = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced)), ref=np.max)
            librosa.display.specshow(D_enh, sr=self.sample_rate, x_axis='time', y_axis='hz', ax=axes[1, 1])
            axes[1, 1].set_title('Enhanced Spectrogram')
        else:
            axes[0, 1].text(0.5, 0.5, 'No speech detected', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Enhanced Waveform')
            axes[1, 1].text(0.5, 0.5, 'No speech detected', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Enhanced Spectrogram')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=150)
        plt.close()

class RealTimePredictor:
    def __init__(self, model, feature_extractor):
        self.model = model
        self.feature_extractor = feature_extractor
        self.model.eval()

    def predict(self, audio_chunk):
        if len(audio_chunk) == 0:
            return -1, 0.0
            
        waveform = torch.from_numpy(audio_chunk.copy()).float().unsqueeze(0)
        features = self.feature_extractor.transform(waveform, MIC_SAMPLE_RATE)
        features = features.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = self.model(features)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            return predicted_idx.item(), confidence.item()

def main():
    print("\n--- Loading Models for Inference ---")
    cnn_model = CNNModel(**CNN_CONFIG).to(DEVICE)
    cnn_model.load_model()
    rnn_model = RNNModel(**RNN_CONFIG).to(DEVICE)
    rnn_model.load_model()
    print("Models loaded successfully.")

    feature_extractor = AudioFeatureExtractor()
    cnn_predictor = RealTimePredictor(cnn_model, feature_extractor)
    rnn_predictor = RealTimePredictor(rnn_model, feature_extractor)
    audio_processor = AudioProcessor()
    
    try:
        while True:
            original_audio = audio_processor.record_audio()
            if original_audio is None: continue

            cleaned_audio = audio_processor.process_audio(original_audio)

            base_path = audio_processor.results_dir / "latest"
            plot_path = base_path.with_suffix('.png')
            original_wav_path = audio_processor.results_dir / "latest_original.wav"
            cleaned_wav_path = audio_processor.results_dir / "latest_cleaned.wav"

            sf.write(original_wav_path, original_audio, audio_processor.sample_rate)
            if len(cleaned_audio) > 0:
                sf.write(cleaned_wav_path, cleaned_audio, audio_processor.sample_rate)

            audio_processor.create_comparison_plot(original_audio, cleaned_audio, plot_path)
            
            cnn_pred, cnn_conf = cnn_predictor.predict(cleaned_audio)
            rnn_pred, rnn_conf = rnn_predictor.predict(cleaned_audio)

            print("\n--- Prediction Results ---")
            print(f"CNN Model:      Predicted {cnn_pred if cnn_pred != -1 else 'N/A'} with {cnn_conf:.2%} confidence")
            print(f"RNN Model:      Predicted {rnn_pred if rnn_pred != -1 else 'N/A'} with {rnn_conf:.2%} confidence")
            print("-" * 26)

            if cnn_pred == -1:
                print("Final Result: No speech detected.")
            elif cnn_conf > rnn_conf:
                final_pred, final_conf, best_model = cnn_pred, cnn_conf, "CNN"
                print(f"Final Result:   {final_pred} (from {best_model} with {final_conf:.2%} confidence)")
            else:
                final_pred, final_conf, best_model = rnn_pred, rnn_conf, "RNN"
                print(f"Final Result:   {final_pred} (from {best_model} with {final_conf:.2%} confidence)")
            
            print(f"\nAudio analysis plot saved to '{plot_path}'")
            print(f"Original audio saved to '{original_wav_path}'")
            if len(cleaned_audio) > 0:
                print(f"Cleaned audio saved to '{cleaned_wav_path}'")

            user_input = input("\nPress Enter to continue, or 'q' to quit: ")
            if user_input.lower() == 'q':
                break

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        print("Program terminated.")


if __name__ == '__main__':
    main()
