# Mask R-CNN for Object Detection and Segmentation
This is the implementation of the "Identification of CDX2 differential expression along the Edapithelium cells of Colon Crypt using Boolean Relationships and deep neural networks on histopathology images" paper. In this project we used python 3, Keras and tensorflow to generate bounding boxes and segmantation masks around each instance. This code was used to train on the gland images from colon tissue and to predict any crypts/glands in the that tissue. 

In this page you find how to reproduce the results from the paper while you will learn how to run it on the other dataset.
For further information about the codebase you can also look at the Matterport repo (https://github.com/matterport/Mask_RCNN).


# Training on Colon Crypts Dataset
In order to train the model on the Colon Crypts dataset, make sure you followed the Installaton process and you have already downloaded the dataset.

```
cd mrcnn/
python3 my_train_crypt.py -h
usage: my_train_crypt.py [-h] --dataset DATASET --dest DEST [--model MODEL]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  path to the dataset, Exp: dataset/Normalized_Images
  --dest DEST        name of the output model, Exp:final.h5
  --model MODEL      path to the model, Exp:
                     logs/no_transfer/mask_rcnn_crypt_0060.h5
 
# Train a new model from the scratch
python3 my_train_crypt.py --dataset dataset/Normalized_Images --dest final.h5

# Train a new model starting from pretrained model
python3 my_train_crypt.py --dataset dataset/Normalized_Images --dest final.h5 --model logs/base_model.h5

```
# Prediction
For model prediction on the given dataset, you can use the following instructions:
```
python my_inference.py -h
usage: my_train_crypt.py [-h] --dataset DATASET --model MODEL

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  path to the dataset, Exp: dataset/Normalized_Images/
  --model MODEL      name of the model, Exp:final.h5

# Prediction and generate the mask files
python3 my_train_crypt.py --dataset=dataset/Normalized_Images/test/ --model=final.h5

```
## Citation
Use this bibtex to cite this paper and repo:
```
@misc{colon_maskrcnn_2020,
  title={Identification of CDX2 differential expression along the Edapithelium cells of Colon Crypt using Boolean Relationships and deep neural networks on histopathology images },
  author={Mahdi Behroozikhah, Soni Khandelwal, Yu Shen, Jaspreet Kaur, Sarah Dabydeen, Sonia Ramamoorthy, Pradipta Ghosh, Soumita Das and Debashis Sahoo},
  year={2021},
}
```
Also this repo is forked from the https://github.com/matterport/Mask_RCNN project and we made few changes through the repo to customize it for the colon crypts detection.

## Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in `requirements.txt`

## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
4. Download the dataset from this link: https://ucsdcloud-my.sharepoint.com/:f:/g/personal/dsahoo_ucsd_edu/EpwRsAH91HxFoj8FTnqh0NUB93MYYdji0hSWaDuMlBMaVQ?e=IKIZ2k and put it in the mrcnn/dataset folder
