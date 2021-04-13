import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import seaborn as sns
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
import argparse

## SET UP CONFIGURATION
parser = argparse.ArgumentParser("color_detection.py")
parser.add_argument("--src", help="path to the dataset, Exp: dataset/Normalized_Images/", type=str, required=True)
parser.add_argument("--dest", help="path to the final result text file, Exp: res.txt. The result table has the following columns in the tsv format: Folder_name, average of the blue intensity in the first 50%% of the image, average blue in the last 50%%, blue in the first 25%%, blue in the last 25%%, followed by the same column for average brown intensity", type=str, required=True)
args = parser.parse_args()

# path = 'dataset/Non_cdx/'"S.19.4704/", "S.19.6308/", "S.19.11919/", "S.19.14180/", "S.19.26681/",
for folder in os.listdir(args.src):
    path = args.src + folder + "/bottom_up/"
    try:
        os.mkdir(path + "blue_brown_mask")
        os.mkdir(path + "mask_and")
    except:
        pass
    # im_names = ['im1', 'im2', 'im3', 'im5', 'im7', 'im8', 'im10', 'im11', 'im15', 'im16']
    im_names = glob.glob(path + '*.png')
    img_size = (400, 1200)
    all_blue = np.zeros(img_size[1])
    all_brown = np.zeros(img_size[1])
    for im_name in im_names:

        def show_image(im1, im2, im3, im4, labels):
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(8, 3))
            ax1.imshow(im1)
            ax2.imshow(im2)
            ax3.imshow(im3)
            ax4.imshow(im4)
            ax1.set_title(labels[0])
            ax2.set_title(labels[1])
            ax3.set_title(labels[2])
            ax4.set_title(labels[3])
            plt.savefig(path + 'blue_brown_mask/' + os.path.basename(im_name) + '.png')
            plt.show()

        def show_image_plt(im1, im2, im3, im4, blue_count, brown_count, labels):
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, figsize=(8, 3))
            ax1.imshow(im1)
            ax1.axis('off')
            ax2.imshow(im2)
            ax2.axis('off')
            ax3.imshow(im3)
            ax3.axis('off')
            ax4.imshow(im4)
            ax4.axis('off')
            ax1.set_title(labels[0])
            ax2.set_title(labels[1])
            ax3.set_title(labels[2])
            ax4.set_title(labels[3])
            ax5.set_title(labels[4])
            ax5.axis('off')

            ax5.plot(blue_count[::-1], range(len(blue_count[::-1])))
            ax5.plot(brown_count[::-1], range(len(brown_count[::-1])))
            # ax5.legend(['blue', 'brown'])
            # ax5.set_xlabel('color value')
            # ax5.set_ylabel('#row')

            fig.savefig(path + 'blue_brown_mask/' + os.path.basename(im_name) + '.pdf', bbox_inches='tight')
            plt.show()

        def mask_builder(image, hl, hh, sl, sh, vl, vh):
            # load image, convert to hsv
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # set lower and upper bounds of range according to arguements
            lower_bound = np.array([hl, sl, vl], dtype=np.uint8)
            upper_bound = np.array([hh, sh, vh], dtype=np.uint8)
            return cv2.inRange(hsv, lower_bound, upper_bound)


        # for mask_name in glob.glob(path+'/Annotation/' + os.path.splitext(os.path.basename(im_name))[0] + '_*'):
        img = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)
        # img = skimage.io.imread(im_name)
        # mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
        # mask = np.array([[max(x) for x in y] for y in mask])
        # img = cv2.bitwise_and(img, img, mask=mask)
        img = cv2.resize(img, img_size)
        img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # nuclei_mask = mask_builder(img, 0, 255, 13, 255, 0, 213)
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thr, _ = cv2.threshold(gray[gray!=0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        nuclei_mask = np.logical_and(0 < gray, gray <= thr)
        opening = cv2.morphologyEx(nuclei_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)


        # show_image(gray, blur)
        # cv2.open nuclei_mask
        # cv2.imwrite(path + 'nuclei_mask/' + os.path.basename(im_name), closing)
        img = cv2.bitwise_and(img, img, mask=closing)
        cv2.imwrite(path + 'mask_and/'+ os.path.basename(im_name) + '.png', img)
        # img = cv2.bitwise_and(img, img, mask=closing)
        # img = cv2.bitwise_and(img, img, mask=th3)
        # cv2.imwrite('tmp1.png', th3)
        # cv2.imwrite('tmp2.png', img)
        # hls = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # nonzero_idx = hls[:, :, 0] != 0
        # sns.distplot(hls[:, :, 0][nonzero_idx])
        # plt.show()
        # thr, _ = cv2.threshold(hls[:, :, 0][nonzero_idx], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # brown_mask = np.logical_and(0 < hls[:, :, 0], hls[:, :, 0] <= thr)
        # blue_mask = hls[:, :, 0] > thr
        brown = mask_builder(img, 0, 40, 1, 254, 1, 254)
        brown = np.logical_or(mask_builder(img, 150, 180, 1, 254, 1, 254), brown) * 255
        blue = mask_builder(img, 80, 140, 1, 254, 1, 254)
        # cv2.imshow('Window', blue)
        # cv2.waitKey(0)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # show_image(img_rgb, brown)

        brown_count = [sum(row == 255) for row in brown]
        blue_count = [sum(row == 255) for row in blue]


        brown_count = gaussian_filter1d(brown_count, sigma=15)
        blue_count = gaussian_filter1d(blue_count, sigma=15)
        blue_count, brown_count = blue_count / (blue_count + brown_count + .00001), brown_count / (blue_count + brown_count + .00001)
        show_image_plt(img_original, img_rgb, blue, brown, blue_count, brown_count, labels=["original", "nuclei_mask", "blue", "brown", "color pattern"])
        # plt.plot(blue_count[::-1], range(len(blue_count[::-1])))
        # plt.plot(brown_count[::-1], range(len(brown_count[::-1])))
        # plt.legend(['blue', 'brown'])
        # plt.savefig(path + 'plots/' + os.path.basename(im_name)+'_res.png')
        # plt.show()
        all_brown += brown_count
        all_blue += blue_count


    blue_percent = np.array([sum(all_blue[0:int(img_size[1]/2)]), sum(all_blue[int(img_size[1]/2):])])
    brown_percent = np.array([sum(all_brown[0:int(img_size[1]/2)]), sum(all_brown[int(img_size[1]/2):])])

    blue_val = blue_percent / (blue_percent + brown_percent)
    brown_val = brown_percent / (blue_percent + brown_percent)

    all_brown = gaussian_filter1d(all_brown, sigma=10)
    all_blue = gaussian_filter1d(all_blue, sigma=10)
    with open(args.dest,'a') as f:
        f.write('{}\t{}\t'.format(folder, len(im_names)))
        for val in [all_blue, all_brown]:
            f50 = np.mean(val[:(len(val) // 2)])
            l50 = np.mean(val[(len(val) // 2):])
            f25 = np.mean(val[:(len(val) // 4)])
            l75 = np.mean(val[(3 * len(val) // 4):])
            f.write('{}\t{}\t{}\t{}\t'.format(f50, l50, f25, l75))
        f.write('\n')
    plt.figure()
    plt.plot(all_blue[::-1], range(len(all_blue[::-1])))
    plt.plot(all_brown[::-1], range(len(all_brown[::-1])))
    plt.legend(['blue', 'brown'])
    plt.xlabel('color value')
    plt.ylabel('#row')
    # plt.plot(all_blue[::-1])
    # plt.plot(all_brown[::-1])
    # plt.legend(['blue', 'brown'])
    plt.savefig(path + 'blue_brown_mask/' +'all_in_one.pdf', bbox_inches='tight')
    plt.show()


print(blue_val)
print(brown_val)