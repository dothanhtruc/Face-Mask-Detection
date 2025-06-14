﻿# Face Mask Detection using PyTorch

This project aims to build a facial mask recognition system through machine learning using PyTorch. The system consists of 2 main parts:

1. **Model Training:** Use image data of faces with and without masks to train CNN.
2. **Inference application:** Use the trained model to recognize mask wearing through 3 inputs:

- Webcam (Real-time detection)

- Still image (Image detection)

- Video (Video detection)

## Objectives

- Build a simple CNN model with PyTorch to classify images of faces with and without masks.

- Apply the model to real-time recognition scenarios using webcam, still images, or videos.

## Features

- **Model training:**
Train the CNN model using data organized by folder:

- `dataset/with_mask`
- `dataset/without_mask`

- **Prediction via webcam:**
Real-time mask recognition via webcam with OpenCV and display results immediately.

- **Prediction via still images and videos:**
Run the model on a saved image or video. The results will be displayed with bounding boxes and labels “Mask” or “No Mask”.

## Directory Structure

> **Note:** If you split the model definition (e.g. class `SimpleCNN`) into a separate file (e.g. `train_model.py`), make sure the import path is set up correctly.

## Requirements

- Python 3.7 or higher
- [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/stable/index.html)
- [OpenCV](https://opencv.org/) (opencv-python)
- [Pillow](https://pillow.readthedocs.io/en/stable/) (for image processing)
- Other libraries: numpy, matplotlib (if plotting is needed)

## Installation

1. **Create virtual environment (optional):**

     conda create -n face_mask_env python=3.8
      conda activate face_mask_env
  
3. **Install required packages:**

   pip install torch torchvision opencv-python pillow numpy
   
4. **Train the model::**

    Open the file train_model.py (or Jupyter Notebook if you use ipynb) and run the entire code to train the model.

    Once training is complete, a face_mask_detector.pth file will be generated containing the model's weights.

> Note: Parameters such as num_epochs, batch_size, learning_rate and input_size in the train_model.py file can be adjusted according to your hardware configuration and requirements.
> 
4. **Run prediction::**

   python main.py

When running, you will be given the option to choose the mode:

1: Recognition via webcam (real-time).

2: Recognition from still images.

3: Recognition from videos.
