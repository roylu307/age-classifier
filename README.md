# age-classifier


## Overview

A simple classifier for identifying age groups from facial images. The sample model is based on a pre-trained EfficientNet [1] model and trained with a newly added classification head. The classification head consists of a global average pooling layer and 1 fully connected layer with [256] channels and an output layer with 6 channels. 1 Dropout layer is used after the first and second fully connected layers with a 0.3 dropout rate. ReLU activation is used for all layers, except the output layer which uses a Softmax function for a classifier.

### Data Pre-processing

The original data contains (48, 48, 1) matrices for images in grayscale. The original images are fed to the network after being converted to RGB format and resized, leading to (128, 128, 3) matrices. From observation, a vast number of images contain black edges, which are caused by background, hair, and illumination conditions. Some of the edges cover a large portion of the image.

Therefore, random translation and rotation are added to the data augmentation stage. The outside boundaries of the augmented images are filled with black (pixel value = 0). The pixel values are then rescaled to [-1, 1] to reduce the magnitude of the values for faster convergence. To further increase the effective number of training samples, the samples are flipped horizontally randomly.

Observations also show that the brightness of the samples is not consistent. An augmentation of random brightness can potentially be added. However, due to some incompatibility issues between the Tensorflow version and the Windows system, layers.RandomBrightness(0.2) is commented out. Besides, the random brightness augmentation is believed to be beneficial to the final results.

Ages are divided into 6 groups ['0-5', '6-18', '19-30', '31-45', '46-65', '66+'] based on the similarities in facial characteristics. However, further considerations can be investigated for further improvements.

### Model Building and Training

The model is built with a backbone of EfficientNet [1] for generating features and a classification head for producing classifier output. The EfficientNet model is a convolutional network with an emphasis on faster training speed and better parameter efficiency, which is a perfect tradeoff between performance and efficiency. I chose the EfficientNetV2S variation as its capacity is suitable for the given dataset size and complexity.

The ADAM optimizer is used for the training with a 0.001 base learning rate. The numbers of channels of layers have been adjusted for optimal performance.


## Model


|               Model               | Accuracy | link |
|:---------------------------------------------:|:-------:|:-------:|
| ckpt-705 |   70.5% | [link](https://drive.google.com/file/d/1KpEdewtyf-KrytVreiyVmxKgJ2nO7cBT/view?usp=drive_link) |





### Environment tested

Our released implementation is tested on.
+ Python 3.9.7 
+ Tensorflow 2.8.3
+ Numpy 1.19.2


### Dataset and Trained Model

For dataset, put age_gender.csv under the root folder.
For trained weights, place the checkpoint file under checkpoint folder
```
age-classifier
├── age_gender.csv
├── checkpoint
│   ├── ckpt-705.h5
```


### Installation

```
cd age-classifier
pip install -r requirements.txt
```

### Training and Testing

#### Testing

```
python val.py 
```

The script will display the images and give the predictions.


#### Training

```
 python train.py
```  

## Reference

[1] Tan, M. and Le, Q., 2021, July. Efficientnetv2: Smaller models and faster training. In International conference on machine learning (pp. 10096-10106). PMLR.
