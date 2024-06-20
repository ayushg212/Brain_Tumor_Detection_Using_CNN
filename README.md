# Brain Tumor Detection using CNN

## Description
This project involves detecting tumors in MRI images using a Convolutional Neural Network (CNN).

## Dataset
The dataset consists of 253 MRI images, with 155 images labeled as tumorous and 98 images labeled as non-tumorous.

## Pre-processing
Pre-processing steps include data augmentation, image resizing, and skull stripping.

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


