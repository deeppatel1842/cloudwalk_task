import os
import torch
import torchaudio
import zipfile
import requests
from torch.utils.data import Dataset
from tqdm import tqdm
from config import DATA_DIR, DATASET_URL, RECORDINGS_PATH, SAMPLE_RATE

def download_and_extract_fsdd():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if os.path.exists(RECORDINGS_PATH):
        print("Dataset already downloaded and extracted.")
        return

    zip_path = os.path.join(DATA_DIR, "fsdd.zip")
    print("Downloading FSDD dataset...")
    try:
        response = requests.get(DATASET_URL, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                size = f.write(chunk)
                bar.update(size)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(zip_path)
        print("Dataset ready.")
    except (requests.exceptions.RequestException, zipfile.BadZipFile) as e:
        print(f"Error during dataset setup: {e}")


class SpokenDigitDataset(Dataset):
    def __init__(self, audio_feature_extractor):
        download_and_extract_fsdd()
        self.feature_extractor = audio_feature_extractor
        self.file_paths = [os.path.join(RECORDINGS_PATH, f) for f in os.listdir(RECORDINGS_PATH) if f.endswith('.wav')]
        self.labels = [int(os.path.basename(f).split('_')[0]) for f in self.file_paths]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        audio_path = self.file_paths[index]
        label = self.labels[index]
        waveform, sr = torchaudio.load(audio_path)
        
        features = self.feature_extractor.transform(waveform, sr)
        return features, label
