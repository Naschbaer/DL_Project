#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 00:25:50 2018

@author: granchgen
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import pickle

import os

from random import randint

def dataset_import(imgf, labelf, n, mappingf = None, reimport=True):
    file_name = imgf + '.pickle'
    images = []
    if not reimport:
        #load previously imported data
        try:
            with open(file_name, 'rb') as f:
                images = pickle.load(f)
                print("dataset_import(): Data imported form cache")
                return char_dict
        except OSError:
            print("dataset_import(): Could not load from chache")
       
    f = open(imgf, "rb")
    l = open(labelf, "rb")
    f.read(16)
    l.read(8)
    
    try:
        k = 0
        while k<n:
            k+=1
            image_label = ord(l.read(1))
            image = np.zeros((28,28))
            for i in range(28):
                for j in range(28):
                    image[j,i] = ord(f.read(1))
            images.append([image_label, image])
            print("Status: %.2f %%" % (float(k)/float(n)*100))
            
        if mappingf is not None:
            mapping = np.genfromtxt (mappingf, delimiter=" ")
            for img in images:
                img[0] = chr(int(mapping[mapping[:, 0]==img[0]][0][1]))
    except TypeError:
        pass
    #save to pickle file
#    try:
#        with open(file_name, 'wb') as f:
#            pickle.dump(images, f, pickle.HIGHEST_PROTOCOL)
#    except OSError:
#        print("dataset_import(): Could not save import")
        
    return images

def image_selection(images):
    num_images = len(images)
    left_image = images[randint(0,num_images-1)][1]
    right_image = images[randint(0,num_images-1)][1]
    idx = randint(0,num_images-1)
    center_image = images[idx][1]
    label = images[idx][0]
    return label, left_image, center_image, right_image

def overlap(left_image, center_image, right_image, 
            overlap = [0.0, 0.0], overlap_begin="border"):
    pixels_left = math.ceil(overlap[0]*28)
    pixels_right = math.ceil(overlap[0]*28)
    
    #make deep copy
    overlap_image = copy.deepcopy(center_image)
    
    if pixels_left>0:
        left_image = np.pad(left_image[:,-pixels_left:], ((0,0),(0,left_image.shape[1]-pixels_left)), mode='constant', constant_values=0)
#        overlap_image += left_image
        overlap_image = np.maximum.reduce([overlap_image,left_image])
    if pixels_right>0:
        right_image = np.pad(right_image[:,:pixels_right], ((0,0),(right_image.shape[1]-pixels_right,0)), mode='constant', constant_values=0)
#        overlap_image += right_image
        overlap_image = np.maximum.reduce([overlap_image,right_image])
    
    # ceil values
#    overlap_image[overlap_image>255] = 255
    return overlap_image
    
def plot(images):
    num_images = len(images)
    y = math.ceil(math.sqrt(num_images))
    x = math.ceil(num_images/y)
    
    f, axarr = plt.subplots(y, x, sharex='col', sharey='row')
    
    for i in range(y):
        for j in range(x):
            if i*x+j >= num_images:
                break
            axarr[i,j].imshow(images[x*i+j][1], cmap='hot', interpolation='nearest')
    plt.show()
    
def to_dictionary(images=None):
    file_name = 'to_dictionary_images.pickle'
    char_dict = {}
    if images is not None:
        for asci in range(97,123):
           character = chr(asci)
           imgs = []
           for img in images: 
               if img[0]==str(character):
                   imgs.append(img[1])
           char_dict[character] = copy.deepcopy(imgs)
         
        for asci in range(65,91):
           character = chr(asci)
           imgs = []
           for img in images: 
               if img[0]==str(character):
                   imgs.append(img[1])
           char_dict[character] = copy.deepcopy(imgs) 
           
        for asci in range(48,58):
           character = chr(asci)
           imgs = []
           for img in images: 
               if img[0]==str(character):
                   imgs.append(img[1])
           char_dict[character] = copy.deepcopy(imgs)
           
        #save the dictionary
        try:
            with open(file_name, 'wb') as f:
                pickle.dump(char_dict, f, pickle.HIGHEST_PROTOCOL)
        except OSError:
            print("to_dictionary(): Could not save dictionary")
    else:
     
        #load previously computed data to save time
        try:
            with open(file_name, 'rb') as f:
                char_dict = pickle.load(f)
                print("to_dictionary(): Data imported form cache")
                return char_dict
        except OSError:
            print("to_dictionary(): Could not load from chache")
            raise ValueError("No input given")
    
    return char_dict
    
if __name__ == '__main__':
#    images = dataset_import("EMNIST/emnist-byclass-train-images-idx3-ubyte", "EMNIST/emnist-byclass-train-labels-idx1-ubyte", 800000, "EMNIST/emnist-byclass-mapping.txt")
#    char_dict = to_dictionary(images)
#    char_dict = to_dictionary()
    path = "/home/granchgen/Dokumente/TUDelf_2nd-Q/Deep_Learning/Project/EMNIST"
    fd = open(os.path.join(path, 'emnist-byclass-train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trainX = loaded[16:].reshape((697932, 28, 28, 1)).astype(np.float32)
#    overlap_images = []
#    for n in range(4):
#        label, left_image, center_image, right_image = image_selection(images)
#        overlap_images.append([label, overlap(left_image, center_image, right_image, [0.3,0.3])])
#    
#    plot(overlap_images)
    
    
    
#    csv = np.genfromtxt ('light_emnist.csv', delimiter=",")
#    labels = csv[:,0]
#    characters = csv[:,1:]
#    car_1 = np.reshape(characters[4,:], (28,28))
#    print (labels)
#   
#    a = np.random.random((16, 16))
#    plt.imshow(car_1.T, cmap='hot', interpolation='nearest')
#    plt.show()