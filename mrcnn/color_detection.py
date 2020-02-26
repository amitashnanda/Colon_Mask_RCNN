import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import glob

from scipy.ndimage.filters import gaussian_filter1d


# path = 'dataset/Non_cdx/'
path = 'dataset/new_data/'

# im_names = ['im1', 'im2', 'im3', 'im5', 'im7', 'im8', 'im10', 'im11', 'im15', 'im16']
im_names = glob.glob(path + '/Images/*')
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

    def mask_builder(image, hl, hh, sl, sh, vl, vh):
        # load image, convert to hsv
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # set lower and upper bounds of range according to arguements
        lower_bound = np.array([hl, sl, vl], dtype=np.uint8)
        upper_bound = np.array([hh, sh, vh], dtype=np.uint8)
        return cv2.inRange(hsv, lower_bound, upper_bound)


    for mask_name in glob.glob(path+'/Annotation/' + os.path.splitext(os.path.basename(im_name))[0] + '_*'):
        img = cv2.imread(im_name)
        mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
        mask = np.array([[max(x) for x in y] for y in mask])
        img = cv2.bitwise_and(img, img, mask=mask)
        # img = cv2.resize(img, img_size)

        nuclei_mask = mask_builder(img, 0, 255, 13, 255, 0, 213)
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(nuclei_mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # show_image(gray, blur)
        # ret3, th3 = cv2.threshold(blur, 50, 100, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.open nuclei_mask
        img2 = cv2.bitwise_and(img, img, mask=nuclei_mask)
        # cv2.imwrite('tmp.png', img2)
        # cv2.imwrite('open.png', cv2.bitwise_and(img, img, mask=opening))
        cv2.imwrite('generated_mask/' + os.path.basename(mask_name), closing)
        img = cv2.bitwise_and(img, img, mask=closing)
        cv2.imwrite('mask_and/'+ os.path.basename(mask_name) + '.png', img)
        continue
        img = cv2.bitwise_and(img, img, mask=closing)
        # img = cv2.bitwise_and(img, img, mask=th3)
        # cv2.imwrite('tmp1.png', th3)
        # cv2.imwrite('tmp2.png', img)
        brown = mask_builder(img, 0, 59, 1, 254, 1, 254)
        blue = mask_builder(img, 125, 227, 1, 254, 1, 254)

        show_image(blue, brown)

        brown_count = [sum(row == 255) for row in brown]
        blue_count = [sum(row == 255) for row in blue]

        all_blue += blue_count
        all_brown += brown_count

        brown_count = gaussian_filter1d(brown_count, sigma=6)
        blue_count = gaussian_filter1d(blue_count, sigma=6)
        plt.plot(blue_count[::-1])
        plt.plot(brown_count[::-1])
        plt.legend(['blue', 'brown'])
        plt.savefig('simple_analysis/plots/' + im_name+'_res.png')
        plt.show()


blue_percent = np.array([sum(all_blue[0:int(img_size[1]/2)]), sum(all_blue[int(img_size[1]/2):])])
brown_percent = np.array([sum(all_brown[0:int(img_size[1]/2)]), sum(all_brown[int(img_size[1]/2):])])

blue_val = blue_percent / (blue_percent + brown_percent)
brown_val = brown_percent / (blue_percent + brown_percent)
all_brown = gaussian_filter1d(all_brown, sigma=15)
all_blue = gaussian_filter1d(all_blue, sigma=15)
plt.plot(all_blue[::-1])
plt.plot(all_brown[::-1])
plt.legend(['blue', 'brown'])
plt.savefig('simple_analysis/plots/' +'all_in_one.png')
plt.show()


print(blue_val)
print(brown_val)