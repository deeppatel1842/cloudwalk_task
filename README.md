Spoken Digit Recognition with PyTorch
This project is a complete system for recognizing spoken digits (0-9) using deep learning. It features two distinct models—a Convolutional Neural Network (CNN) and a Bidirectional Recurrent Neural Network (RNN/LSTM)—and includes a real-time inference pipeline with advanced audio processing capabilities.

Key Features
Dual-Model Architecture: Implements both a CNN for spatial feature recognition in spectrograms and a Bidirectional RNN (LSTM) for understanding temporal sequences in audio.

Advanced Audio Processing: The real-time inference pipeline includes a multi-stage cleaning process:

High-pass filtering to remove low-frequency noise.

Noise reduction using spectral gating.

Voice Activity Detection (VAD) to trim silence.

Normalization of duration and loudness for consistent input.

Real-Time Inference: A user-friendly script (microphone_inference.py) allows for live testing of the models using a microphone. It provides predictions from both models and a final "best guess" based on confidence scores.

Detailed Analysis & Visualization: The system automatically generates and saves detailed comparison plots, showing the original vs. cleaned audio waveforms and spectrograms.

Robust Training Pipeline: The training script (train.py) includes best practices like learning rate scheduling and early stopping to prevent overfitting and ensure the best possible model is saved.

Project Structure
The project is organized into a flat structure for simplicity, with each file having a distinct purpose.

.
├── config.py                   # Central configuration for all parameters
├── dataset_loader.py           # Handles downloading and loading the FSDD dataset
├── audio_feature_extraction.py # Converts raw audio to Mel Spectrograms
├── models.py                   # Defines the CNN and RNN model architectures
├── train.py                    # Script to train both models
├── evaluation.py               # Script to evaluate the trained models on the test set
├── microphone_inference.py     # Main script for real-time prediction
├── audio_recorder_analyzer.py  # Standalone tool for audio recording and analysis
├── problem.md                  # Documentation of challenges and solutions
├── requirements.txt            # Required Python packages
└── README.md                   # This file

Setup and Installation
Follow these steps to set up your environment and run the project.

1. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

python -m venv fsdd_env
source fsdd_env/bin/activate  # On Windows, use `fsdd_env\Scripts\activate`

2. Install Dependencies
Install all the required packages using the requirements.txt file.

pip install -r requirements.txt

Note: The requirements.txt file should contain the following packages:

torch
torchaudio
sounddevice
numpy
scipy
matplotlib
librosa
noisereduce
webrtcvad-wheels
pandas
seaborn
tqdm
requests
tensorboard

How to Run
The project is divided into a training phase and an inference phase. You must run them in order.

Step 1: Train the Models
This script will automatically download the dataset, train both the CNN and RNN models, and save the best-performing versions in the saved_models directory.

python train.py

Step 2: Evaluate the Trained Models
After training, you can run this script to evaluate the models' performance on the unseen test set. It will generate classification reports and confusion matrices in the results folder.

python evaluation.py

Step 3: Run Real-Time Inference
This is the main application. It will load both trained models and allow you to make live predictions using your microphone.

python microphone_inference.py

The script will save the recorded audio and a detailed analysis plot in the live_recordings folder.

(Optional) Standalone Audio Analyzer
If you only want to test the audio recording and cleaning pipeline without running the prediction models, you can use this script.

python audio_recorder_analyzer.py

Key Files Explained
config.py: The central hub for all project settings. Adjust parameters like learning rate, batch size, and audio processing settings here.

models.py: Contains the PyTorch class definitions for the CNNModel and RNNModel (Bidirectional LSTM).

train.py: Manages the entire training and validation loop, including early stopping to prevent overfitting.

microphone_inference.py: The core of the real-time application. It handles audio recording, processing, dual-model prediction, and user interaction.

problem.md: A detailed log of the technical challenges faced during development and the solutions that were implemented to overcome them.