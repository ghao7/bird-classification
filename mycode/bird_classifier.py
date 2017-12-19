# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import skimage
import sys
import matplotlib.pyplot as plt
from nms_fast import find_non_max
import matplotlib.patches as mpatches
import selectivesearch
import numpy as np
import cv2
import os
import classifier.src.label_image as my_classifier

def main():
    image_path = sys.argv[1]
    path_list = image_path.split('.')
    im_path = path_list[0]
    file = image_path

    if not os.path.exists(im_path):
        os.makedirs(im_path)
    # loading astronaut image
    # img = skimage.data.astronaut()
    img = skimage.io.imread(file)
    # cv2.imshow('im',img[100:200, 300:400])
    # cv2.waitKey(0)

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)


    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 3000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)

    images = [(file, np.array(list(candidates)))]
    # print(pick)
    new_candiates = find_non_max(images)
    print(new_candiates)

    print('something')

    j = 0
    for x, y, w, h in new_candiates:
        str_im = 'crop' + str(j)
        path = im_path + '/' + str_im+'.png'
        cv2.imwrite(path,img[y: y + h, x: x + w])
        # cv2.imshow(str_im, img[y: y + h, x: x + w])
        # cv2.waitKey(0)
        result = my_classifier.classify(path)
        if result != 'none':
            if result == 'americanpekin':
                color = 'white'
            if result == 'mallardducks':
                color = 'green'
            if result == 'canadageese':
                color = 'red'
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor= color, linewidth=1)
            ax.add_patch(rect)
            print("find it " + result)
        j = j + 1

    plt.show()

if __name__ == "__main__":
    main()
