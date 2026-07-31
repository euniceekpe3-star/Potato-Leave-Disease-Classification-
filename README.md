 # Potato Leaf Disease Classification

A Convolutional Neural Network (CNN) that classifies potato leaf images as either **Healthy** or affected by **Early Blight**, deployed as an interactive web application using Streamlit.

🔗 **Live App:** https://kzypdujishzchxpf3kgp4t.streamlit.app/

## Overview

This project was developed for GET 324 (Artificial Intelligence / Machine Learning). It covers the full pipeline from model training to public deployment:

1. Training a CNN on a labelled potato leaf image dataset
2. Evaluating the model on a held-out test set
3. Building a Streamlit interface for real-time image upload and prediction
4. Deploying the app to Streamlit Community Cloud via GitHub

## Model

- **Task:** Binary image classification (Healthy vs. Early Blight)
- **Architecture:** 3 Conv2D + MaxPooling blocks (32 → 64 → 128 filters) → Flatten → Dense(128, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
- **Input size:** 128 × 128 RGB images
- **Training:** 15 epochs, Adam optimizer, binary cross-entropy loss
- **Test accuracy:** 100% (on a 95-image test set)

> **Note:** This model was trained only on Healthy and Early Blight classes, per the assignment scope. It has no mechanism to recognize Late Blight or other conditions — any image outside its two trained classes will still be forced into one of them with high confidence, since it uses a binary sigmoid output rather than a multi-class softmax.

## Project Structure

```
├── app.py                          # Streamlit application
├── models/
│   └── potato_blight_classifier.keras   # Trained model
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version pin for deployment
└── .gitignore
```

## Running Locally

```bash
git clone https://github.com/euniceekpe3-star/Potato-Leave-Disease-Classification-.git
cd Potato-Leave-Disease-Classification-
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Tools Used

Python · TensorFlow / Keras · Streamlit · NumPy · Pillow · Git & GitHub · Streamlit Community Cloud

## Authors

- Eunice Bassey Ekpe — 23/EG/CE/030
- Odey Esther — 23/EG/CE/040
- peter, sarah wuwuda- 23/EG/CE/010
- Ulaeto, Able Ene - 23/EG/CE/090
- Edward,Joseph Bassey- 23/EG/CE/020
- Okorie,Chidera Perpetual peace -23/EG/CE/110
- Ekanem Esther Sunday - 23/EG/CE/050
- Uwe,Rhoda Etim - 23/EG/CE/070
