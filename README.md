# Mask R-CNN for Object Detection and Segmentation
This repository details the Python 3 implementation of the research findings from the paper titled "Expression Gradient of Cancer Suppressor Gene Found in Colon Crypt Using Vision-AI". Utilizing Keras and TensorFlow frameworks, the project's objective is to accurately delineate crypts and glands within colon tissue images (histopathological) by creating bounding boxes and segmentation masks for each identified instance. The developed code has been trained and is capable of predicting, the presence of crypts and glands in colon tissue images. 

To replicate the study's outcomes and apply the methodology to additional datasets, this guide offers comprehensive instructions. For those interested in a deeper dive into the code's foundation, the Matterport repository at https://github.com/matterport/Mask_RCNN provides extensive resources and foundational elements pertinent to this work.

## Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in `requirements.txt`.

## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   conda env create -f conda_environment.yml 
   conda activate base
   pip install histomicstk --find-links https://girder.github.io/large_image_wheels
   ```
   ```bash
   pip install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
4. Download the dataset from this link: https://ucsdcloud-my.sharepoint.com/:f:/g/personal/dsahoo_ucsd_edu/EpwRsAH91HxFoj8FTnqh0NUB93MYYdji0hSWaDuMlBMaVQ?e=IKIZ2k and put it in the mrcnn/dataset folder
5. Download the inference dataset from this link: https://ucsdcloud-my.sharepoint.com/:f:/g/personal/dsahoo_ucsd_edu/EgOIJFepWBJFpFIY8TsocsgBT8NxJ8M7DfWHUlWV-Q1bcQ?e=Ljk1cr
6. Add this project to your pythonpath
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/this/project/root"
   ```


# Training on Colon Crypts Dataset
Before training the model on the Colon Crypts and glands dataset, ensure that you have completed the installation process and successfully downloaded the dataset in the correct folder.
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
usage: my_inference.py [-h] --dataset DATASET --model MODEL

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  path to the dataset, Exp: dataset/Normalized_Images/
  --model MODEL      name of the model, Exp:final.h5

# Prediction and generate the mask files
python3 my_inference.py --dataset=dataset/Normalized_Images/test/ --model=final.h5

```
# Running the whole pipeline
To reproduce the result from the paper, you might need to normalize the color spectrum in the dataset using reinhard algorithm. 
```
python color_normalize_pipe.py -h
usage: color_normalize_pipe.py [-h] --ref REF --dest DEST --src SRC

optional arguments:
  -h, --help   show this help message and exit
  --ref REF    path to the reference image, Exp: dataset/color_reference.png
  --dest DEST  Folder path of the output images, Exp:dataset/Color_Normalized/
  --src SRC    Folder path of the source images (will only process png files),
               Exp: dataset/Raw_images/
# Normalize the dastaset images
python3 color_normalize_pipe.py 
```
Having the color normalized data, you need to apply hough transform to better align the images before the machine learning pipeline.
```
usage: hough_transform.py [-h] --dest DEST --src SRC

optional arguments:
  -h, --help   show this help message and exit
  --dest DEST  Root folder path of the dest images Exp: images/
  --src SRC    Root folder path of the source images (will only process png
               files) Exp: if you have images in images/data1/1.png
               images/data2/3.png you should pass: images/
# Transform the dataset
python3 hough_transform.py --src dataset/test_images/ --dest dataset/test_images/ 
```
After the image xy-axis align, now you can run the prediction on the same folder, because the last command will store the images in the "dest_path/images", so you have to pass the same "dest_path" here.
```
python3 my_inference.py --dataset=dataset/test_images/ --model=final.h5
```
Now there will be another folder with the predicted mask of the provided images, in the folder named "predicted_mask". From here you can call the next pipeline to rotate the U-shapes to align them from bottom to top for the futher analysis.
```
python U-shape-bottom-pipeline.py -h
usage: U-shape-bottom-pipeline.py [-h] --src SRC

optional arguments:
  -h, --help  show this help message and exit
  --src SRC   Root folder path of the process images(has to include
              predicted_mask and images) Exp: dataset/test_images/
# Bottom up alignment
python3 U-shape-bottom-pipeline.py --src dataset/test_images/
```
Finally in the last step, the color spectrum needs to get measured from bottom to top. The following script will separate the image_mask, plot the blue vs brown color for each image and also aggregate all the images result together in the final result text file.
```
python color_detection.py -h
usage: color_detection.py [-h] --src SRC --dest DEST

optional arguments:
  -h, --help   show this help message and exit
  --src SRC    path to the dataset, Exp: dataset/Normalized_Images/
  --dest DEST  path to the final result text file, Exp: res.txt. The result
               table has the following columns in the tsv format: Folder_name,
               average of the blue intensity in the first 50% of the image,
               average blue in the last 50%, blue in the first 25%, blue in
               the last 25%, followed by the same column for average brown
               intensity

# Color detection values
python3 color_detection.py --src dataset/test_images/ --dest res.txt
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
