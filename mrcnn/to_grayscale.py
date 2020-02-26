import cv2
import skimage
import glob, os, errno

for fil in glob.glob("dataset/Mixed_crypts/train/Annotation/1005479.svs (1, 5323, 37119, 4149, 6193)*_.png"):
    print(fil)
    image = cv2.imread(fil, 0)
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
    cv2.imwrite(fil, image) # write to location with same name
