# Brain Tumor Detection using CNN

## Table of Contents
- [Description](#description)
-  [Dataset](#dataset)
- [Pre-processing](#pre-processing)
- [Model](#model)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results](#results)

## Description
This project involves detecting tumors in MRI images using a Convolutional Neural Network (CNN).

## Dataset
The dataset consists of 253 MRI images, with 155 images labeled as tumorous and 98 images labeled as non-tumorous.

## Pre-processing
Pre-processing steps include data augmentation, image resizing, and skull stripping.

1. **Data Augmentation**: Data Augmentation is a crucial technique used in this project to enhance the dataset and improve the performance of the Convolutional Neural Network (CNN). Given the relatively small size of the dataset (253 MRI images), data augmentation helps to artificially increase the size and variability of the training data, which can lead to better generalization and robustness of the model.

Data Before Augmentation | Data After Augmentation
:-------------------------:|:-------------------------:
![alt text](https://github.com/ayushg212/Brain_Tumor_Detection_Using_CNN/blob/main/images/dataset_before_augementation.png) |  ![alt text](https://github.com/ayushg212/Brain_Tumor_Detection_Using_CNN/blob/main/images/dataset_after_augmentation.png)

2. **Skull Stripping** : Skull Stripping is an essential pre-processing step used in this project to improve the accuracy and efficiency of the Convolutional Neural Network (CNN) in detecting brain tumors from MRI images.Skull Stripping involves removing non-brain tissues (such as the skull, scalp, and other extraneous structures) from MRI images, leaving only the brain tissue.

Skull Stripping for Non tumourous Image | Skull Stripping for tumourous Image
:-------------------------:|:-------------------------:
![alt text](https://github.com/ayushg212/Brain_Tumor_Detection_Using_CNN/blob/main/images/skull_croping_nontumor_example.png) |  ![alt text](https://github.com/ayushg212/Brain_Tumor_Detection_Using_CNN/blob/main/images/skull_croping_tumor_example.png)

## Model
A pre-trained VGG19 model with a custom fully connected layer was used for this project. The custom fully connected layer was trained to achieve an accuracy of 98%.

## Technologies Used
- Python
- Keras
- TensorFlow
- NumPy
- Pandas
- OpenCV

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/ayushg212/Brain_Tumor_Detection_Using_CNN.git  
2. Navigate to the project directory:
   ```
   cd Brain_Tumor_Detection_Using_CNN
## Usage

1. Run the Jupyter Notebook:
   ```
   jupyter notebook DL_Project_Brain_Tumor_Image_Classification.ipynb
## Results
The model achieved an accuracy of 98% on the validation set.


