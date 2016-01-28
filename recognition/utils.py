import pickle
from PIL import Image
import os
from . import imageDescriptor
import numpy as np
import random
import cv2


# read a folder of images and return their features
def readImages(folderPath, sift):
    images = []

    stackOfFeatures = []
    for label in os.listdir(folderPath):
        if label == '.DS_Store':
            continue
        imagesPath = folderPath +"/"+ label

        for imagePath in os.listdir(imagesPath):
            if imagePath == ".DS_Store":
                continue

            imagePath = imagesPath +"/"+ imagePath
            if sift:
                print "Extract SIFT features of " + imagePath
            else:
                print "Extract Brisk features of " + imagePath

            img = cv2.imread(imagePath)
            height, width, depth = img.shape

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Use sift
            if sift:
                sift_obj = cv2.xfeatures2d.SIFT_create()
                [kps, desc] = sift_obj.detectAndCompute(img_gray, None)
            else:
                # Use brisk
                brisk = cv2.BRISK_create(30, 3, 1.0)
                [kps, desc] = brisk.detectAndCompute(img_gray, None)

            if desc is None:
                desc = []
            numberOfDescriptors = len(desc)
            descriptors = []
            for i in range(numberOfDescriptors):
                descriptor = imageDescriptor.siftDescriptor(kps[i].pt[0], kps[i].pt[1], desc[i])
                descriptors.append(descriptor)
                stackOfFeatures.append(descriptor.descriptor)

            imDescriptors = imageDescriptor.imageDescriptors(descriptors, label, width, height)
            images.append(imDescriptors)

    return [images, stackOfFeatures]
