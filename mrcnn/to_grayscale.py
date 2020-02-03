import cv2
import skimage
import glob, os, errno

for fil in glob.glob("dataset/new_crypts/train/Annotation/*.png"):
    image = cv2.imread(fil, 0)
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
    cv2.imwrite(fil, image) # write to location with same name