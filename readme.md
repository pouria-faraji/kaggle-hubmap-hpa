# HuBMAP + HPA - Hacking the Human Body
This repository contains source code for an image segmentation Kaggle competition, called [**HuBMAP + HPA - Hacking the Human Body**](https://www.kaggle.com/competitions/hubmap-organ-segmentation)

In this competition we have identify and segment functional tissue units (FTUs) across five human organs.

There are three main Jupyter notebooks. Notebookes can easily be run in Google Colab. You have to first download your `kaggle.json` API token file from settings. 

1. **Data Preparation** 

    First, image masks are generated using `rle2mask` function. Second, all 351 images will be divided in groups in 3 sub-folders for train, validation, and test sets. Number of images in validation and test folders will be determinned by `TEST_SIZE` .

    Third, all images will be splitted into patches of `PATCH_SIZE` , then will be augmented by random rotation and random flip. 

2. **Model Training**
    
    Two models have been used, which can be accessed using the corresponding filename. 
    
    One is simple U-Net with *Encoder* and *Decoder* as it's building blocks. The corresponding file is `UNet.py`

    The other is a U-Net with *EfficientNet* as the backbone. For using this model, `segmentation-model` library must be installed using the following command:

    `pip install segmentation-models`

    For optimizer, *Adam* is used, and for the loss, *Binary Cross-Entropy*. The score metric is *Dice Coefficient*. The first 10 epochs the weights of encoders are freezed so they won't change drastically during first epochs of training. Then they are unfreezed for training afterwards.

3. **Results**

    The following tables show the results for a training of a simple U-Net in comparison to the U-Net with Efficientnet as the backbone.

    | Model           | Train Loss | Validation Loss | Test Loss |
    |-----------------|------------|-----------------|-----------|
    | Simple U-Net    | 0.088      | 0.095           | 0.115     |
    | EfficientNet    | 0.039      | 0.125           | 0.169     |

    | Model           | Train Dice | Validation Dice | Test Dice |
    |-----------------|------------|-----------------|-----------|
    | Simple U-Net    | 0.85       | 0.86            | 0.86      |
    | EfficientNet    | 0.93       | 0.81            | 0.91      |

