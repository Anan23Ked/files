# Expressions

This project is a real-time facial expression recognition system built using a machine learning model that processes live camera input. The model was developed and deployed on a Raspberry Pi device and is capable of accurately classifying four emotional states: happy, sad, angry, and neutral. It was trained on a dataset of facial images representing these expressions. Built with TensorFlow, the system produces a physical output by converting the detected expressions into integer values, which are then transmitted to an ESP32 microcontroller to trigger specific hardware interactions.

## Overview
### Who is it for?
A social installation where people can engage with a familiar-looking face that mirrors their own emotions, creating an interactive and immersive experience for users.

### Why does it exist?
This project explores what it means to be human through the lens of artificial intelligence embodied in a physical form. By observing and interpreting human facial expressions in real time, the system learns to mimic and respond to the emotions it perceives, much like a child learning from its surroundings. It exists as an experiment in how AI can move beyond abstract computation into a tangible, interactive presence—reflecting human behavior, emotions, and social cues through both digital intelligence and physical response.

## Features

### Real-Time Facial Expression Classification
Performs inference on live video streams to classify facial expressions into four categories: happy, sad, angry, and neutral.
### Edge Inference on Raspberry Pi
Optimized to run directly on a Raspberry Pi, enabling low-latency, on-device processing without reliance on cloud services.
### TensorFlow-Based Model Architecture
Implements a trained deep learning model using TensorFlow for robust facial emotion recognition.
### Preprocessed Training Pipeline
Model trained on labeled facial image datasets with preprocessing steps such as normalization and resizing to ensure consistent input.
### Emotion Encoding and Serial Transmission
Encodes predicted emotion classes into integer values for efficient communication to external devices.
### ESP32 Hardware Integration
Interfaces with an ESP32 microcontroller to execute predefined hardware responses based on model output.

## Tech Stack

1. Language: Python for ML model developmentand C++ hardware programming
2. Frameworks/Libraries:Tensorflow, TFLite, OpenCV
3. Platform: VSCode, Arduino IDE

## Installation
### 1. Clone the repository
<git clone https://github.com/Anan23Ked/files.git>
<cd files>

### 2. Install dependencies
<pip install -r requirements.txt>

### 3. Usage
<python main.py>
