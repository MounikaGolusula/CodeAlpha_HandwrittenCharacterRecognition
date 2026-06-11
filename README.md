# Handwritten Digit Recognition using CNN

## Project Overview

This project implements a Handwritten Digit Recognition system using a Convolutional Neural Network (CNN) trained on the MNIST dataset. A Streamlit web application allows users to upload handwritten digit images ( 0-9 ) and receive predictions in real time.

## Features

* MNIST dataset preprocessing
* CNN-based digit classification
* Model accuracy evaluation
* Saved trained model
* Streamlit web interface
* Real-time digit prediction

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Streamlit
* Pillow

## Project Structure

handwritten-recognition-system/

├── model/

│ └── mnist_cnn.h5

├── app.py

├── train_model.py

├── requirements.txt

├── README.md

└── .gitignore

## Model Performance

Test Accuracy: ~98-99%

## Installation

```bash
pip install -r requirements.txt
```

Train Model

```bash
python train_model.py
```

Run Application

```bash
streamlit run app.py
```

Dataset

MNIST Handwritten Digits Dataset

Classes: 0–9

Author

Mounika Golusula
