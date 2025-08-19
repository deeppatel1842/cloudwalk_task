Project Challenges and Solutions
This document outlines the main technical challenges faced during the development of the Spoken Digit Recognition project and the corresponding solutions that were implemented.

1. Model Brittleness: Input Shape Mismatch in CNN
Problem
The most significant issue encountered was a RuntimeError during the real-time inference phase. The error, mat1 and mat2 shapes cannot be multiplied, occurred because the CNN model was trained on spectrograms of a fixed size (e.g., 64x32), but the spectrograms generated from live microphone audio sometimes had minor variations in their time dimension (e.g., 64x31 or 64x33). The CNN's final fully connected layers were hardcoded to expect one specific input shape, causing the model to crash when it received anything else.

Solution
The solution was to make the CNN architecture more robust and flexible, removing its dependency on a fixed input length.

Global Average Pooling (GAP): We replaced the rigid nn.Flatten() layer with an nn.AdaptiveAvgPool2d((1, 1)) layer.

How it Works: Instead of flattening a variable-sized feature map, the GAP layer averages the values across each channel, producing a fixed-size output (in this case, a [batch_size, 64, 1, 1] tensor). This consistently sized output could then be safely passed to the final linear layers without causing a shape mismatch.

Outcome: This architectural change permanently solved the error and resulted in a more robust model that could handle slight variations in audio duration, making it much better suited for real-world applications. This required retraining the model to learn new weights compatible with the new architecture.

2. Poor Generalization: Model Overfitting
Problem
During training, we observed that the model's Training Loss continued to decrease while the Validation Loss stagnated or began to increase. This is a classic sign of overfitting, where the model starts to "memorize" the training data instead of learning general patterns. This resulted in poor performance on new, unseen audio.

Solution
The most effective solution was to implement Early Stopping.

How it Works: We monitored the validation loss at the end of each epoch. The training process was configured to automatically terminate if the validation loss did not show any improvement for a set number of consecutive epochs (a "patience" of 5).

Outcome: This ensured that we saved the model at its peak performance, right before it started to overfit. It prevented wasted training time and resulted in a final model that generalized much better to new data, leading to lower and more stable validation loss scores.

3. Inconsistent Real-Time Audio Quality
Problem
The initial microphone recording script produced inconsistent and often noisy audio. Issues included:

Capturing audio before the user started speaking.

Significant background noise interfering with the spoken digit.

Inconsistent volume levels.

Solution
We implemented a multi-stage Advanced Audio Processing Pipeline to clean and standardize the microphone input before feeding it to the models.

Stable Recording: Switched from pyaudio to the sounddevice library for more reliable audio capture.

High-Pass Filter: Applied a filter to remove low-frequency hum.

Noise Reduction: Used the noisereduce library to subtract ambient background noise.

Voice Activity Detection (VAD): Implemented webrtcvad to automatically detect the start and end of speech, trimming away leading and trailing silence.

Normalization: Standardized the duration and loudness of the final trimmed audio clip.

Outcome: This pipeline produced a much cleaner and more consistent audio signal, significantly improving the accuracy and reliability of the real-time predictions.

4. Dependency Issue: torchcodec Missing from Hugging Face datasets
Problem Description
When attempting to train the CNN model using the Free Spoken Digit Dataset (FSDD) from Hugging Face, the following error occurred:

ModuleNotFoundError: No module named 'torchcodec.decoders'

Full Error Traceback:

Error during training: No module named 'torchcodec.decoders'
Traceback (most recent call last):
  File "C:\Users\Kashyap\Documents\Deep\cloudwalk_task\scripts\train_cnn.py", line 99, in <module>
    model, metrics, history = train_cnn_model()
  File "C:\Users\Kashyap\Documents\Deep\cloudwalk_task\scripts\train_cnn.py", line 30, in train_cnn_model
    data_loader = FSSDDataLoader(feature_type='mfcc', batch_size=TRAINING_CONFIG['batch_size'])
  File "C:\Users\Kashyap\Documents\Deep\cloudwalk_task\data\dataset_loader.py", line 94, in __init__
    self._prepare_data_splits()
  File "C:\Users\Kashyap\Documents\Deep\cloudwalk_task\data\dataset_loader.py", line 105, in _prepare_data_splits
    for sample in self.dataset[split]:
  ...
  File "C:\Users\Kashyap\Documents\Deep\cloudwalk_task\fsdd_env\Lib\site-packages\datasets\features\_torchcodec.py", line 2, in <module>
    from torchcodec.decoders import AudioDecoder as _AudioDecoder
ModuleNotFoundError: No module named 'torchcodec.decoders'

Root Cause Analysis
The issue occurred because:

Hugging Face Datasets Library: The datasets library was trying to automatically decode audio files using torchcodec.decoders.AudioDecoder.

Missing Dependency: The torchcodec package was not installed in the environment.

Automatic Audio Decoding: When loading the FSDD dataset, the library automatically attempts to decode audio files during iteration.

Environment Compatibility: torchcodec is not always available or compatible with all PyTorch installations.

Solution Implemented
Instead of relying on the Hugging Face datasets library's automatic decoding, we switched to a more direct and reliable data loading method.

Direct Download from GitHub: The dataset_loader.py script was modified to download the FSDD dataset directly from the original author's GitHub repository as a .zip file.

Manual Audio Loading: The script now manually loads the raw .wav files using torchaudio.load, bypassing the problematic torchcodec dependency entirely.

Outcome: This approach provided a more stable and environment-independent way to load the audio data, completely resolving the ModuleNotFoundError.