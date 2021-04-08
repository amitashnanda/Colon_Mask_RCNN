import histomicstk as htk
import cv2
import os
import matplotlib.pyplot as plt
import argparse

#######################################################################################
## SET UP CONFIGURATION
parser = argparse.ArgumentParser("color_normalize_pipe.py")
parser.add_argument("--ref", help="path to the reference image, Exp: dataset/color_reference.png", type=str, required=True)
parser.add_argument("--dest", help="Folder path of the output images, Exp:dataset/Color_Normalized/", type=str, required=True)
parser.add_argument("--src", help="Folder path of the source images (will only process png files), Exp: dataset/Raw_images/", type=str, required=True)
args = parser.parse_args()

def reinhard_img(directory, des_path):
    img = cv2.imread(args.ref)
    mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(img)

    for filename in os.listdir(directory):
        print(filename)
        if filename.endswith(".png"):
            img_input = cv2.imread(directory + filename)
            plt.imshow(img_input), plt.show()
            im_nmzd = htk.preprocessing.color_normalization.reinhard(img_input, mean_ref, std_ref)
            status = cv2.imwrite(des_path + filename, im_nmzd)
            plt.imshow(im_nmzd), plt.show()
            print("Image written to file-system : ", status)
            continue
        else:
            continue


reinhard_img(args.src, args.dest)
