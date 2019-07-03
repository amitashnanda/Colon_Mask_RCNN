import cv2
import numpy as np
from matplotlib import pyplot as plt

from scipy.ndimage.filters import gaussian_filter1d

im_names = ['im1', 'im2', 'im3', 'im5', 'im7']
img_size = (400, 1200)
all_blue = np.zeros(img_size[1])
all_brown = np.zeros(img_size[1])
for im_name in im_names:

    def show_image(im1, im2):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
        ax1.imshow(im1)
        ax2.imshow(im2)
        ax1.set_title('blue')
        ax2.set_title('brown')
        plt.savefig('simple_analysis/blue_brown_mask/' + im_name + '_blue_brown.png')
        plt.show()

    img = cv2.imread('simple_analysis/images/'+im_name +'.png')

    mask = cv2.imread('simple_analysis/mask/'+im_name +'_mask.png', 0)
    img = cv2.bitwise_and(img, img, mask=mask)
    img = cv2.resize(img, img_size)

    def mask_builder(image, hl, hh, sl, sh, vl, vh):
        # load image, convert to hsv
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # set lower and upper bounds of range according to arguements
        lower_bound = np.array([hl, sl, vl], dtype=np.uint8)
        upper_bound = np.array([hh, sh, vh], dtype=np.uint8)
        return cv2.inRange(hsv, lower_bound, upper_bound)

    brown = mask_builder(img, 0, 59, 1, 254, 1, 254)
    blue = mask_builder(img, 125, 227, 1, 254, 1, 254)

    show_image(blue, brown)

    brown_count = [sum(row == 255) for row in brown]
    blue_count = [sum(row == 255) for row in blue]

    all_blue += blue_count
    all_brown += brown_count

    brown_count = gaussian_filter1d(brown_count, sigma=15)
    blue_count = gaussian_filter1d(blue_count, sigma=15)
    plt.plot(blue_count[::-1])
    plt.plot(brown_count[::-1])
    plt.legend(['blue', 'brown'])
    plt.savefig('simple_analysis/plots/' + im_name+'_res.png')
    plt.show()

all_brown = gaussian_filter1d(all_brown, sigma=15)
all_blue = gaussian_filter1d(all_blue, sigma=15)
plt.plot(all_blue[::-1])
plt.plot(all_brown[::-1])
plt.legend(['blue', 'brown'])
plt.savefig('simple_analysis/plots/' +'all_in_one.png')
plt.show()