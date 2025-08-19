# Spoken Digit Recognition - A Deep Learning Approach

Welcome to the **Spoken Digit Recognition** project!  
This is an **end-to-end system** that uses deep learning to recognize spoken digits (**0–9**) in real-time.  
It features a **robust audio processing pipeline**, a **dual-model prediction engine**, and **visual analysis** for every prediction.

---

## What Makes This Special
This project was designed to tackle the **real-world challenges** of audio classification.  
Instead of only being accurate in perfect conditions, it is **robust against noise, silence, and speech variations**.  

The strength of the system lies in:  
- **Signal Processing + Deep Learning** working together.  
- **Comparative dual-model approach** for reliability.  
- **Immediate, transparent feedback** with plots and confidence scores.  

---

## What I Would Build Next (If I Had 2 More Hours)
1. **Keyword Spotting:** Extend recognition to a small vocabulary (e.g., “yes”, “no”, “stop”, “go”).  
2. **Web-Based Interface:** Add a simple **Flask/FastAPI frontend** so anyone can test the model from their browser.  

---

## Key Features
- **Dual-Model Prediction Engine**  
  Combines a **CNN** and a **Bidirectional RNN (LSTM)** for stronger predictions.  

- **Advanced Audio Processing**  
  Multi-stage pipeline: high-pass filter → noise reduction → silence trim → normalization.  

- **Real-Time Inference**  
  Record, clean, predict, and visualize — all in seconds.  

- **Early Stopping in Training**  
  Prevents overfitting and ensures models generalize to new voices.  

- **Comparative “Best Guess”**  
  Picks the prediction with the **highest confidence score** across models.  

---

## How It Works

### Run the System
```bash
python microphone_inference.py
