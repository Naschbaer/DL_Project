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

import sys

import markov_chain_nl as mcl

from random import randint

def dataset_import(imgf, labelf, n = sys.maxsize , mappingf = None):
    images = []
       
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
    except TypeError:
        pass
            
    if mappingf is not None:
        mapping = np.genfromtxt (mappingf, delimiter=" ")
        for img in images:
            img[0] = chr(int(mapping[mapping[:, 0]==img[0]][0][1]))

        
    return images

def dataset_export(imgf, labelf, dataset, mappingf = None):     
    f = open(imgf, "wb")
    l = open(labelf, "wb")
    f.write(b'\x00\x00\x08\x03\x00\n\xa6L\x00\x00\x00\x1c\x00\x00\x00\x1c')
    l.write(b'\x00\x00\x08\x01\x00\n\xa6L')
    
    mapping = np.genfromtxt (mappingf, delimiter=" ")
   
    k = 0
    n = len(dataset)
    for data in dataset:
        #write label
        if mappingf is not None:
            #data[0] is the label for the tree images and index [1] returns the middle one
            label = int(mapping[mapping[:, 1]==ord(data[0][1])][0][0])
        l.write(label.to_bytes(1, byteorder='big'))
        image = data[1]
        
        for i in range(28):
            for j in range(28):
                f.write(int(image[j,i]).to_bytes(1, byteorder='big'))

        print("Status: %.2f %%" % (float(k)/float(n)*100))
        k+=1


def image_selection(images):
    num_images = len(images)
    left_image = images[randint(0,num_images-1)][1]
    right_image = images[randint(0,num_images-1)][1]
    idx = randint(0,num_images-1)
    center_image = images[idx][1]
    label = images[idx][0]
    return label, left_image, center_image, right_image

def image_selection_markov_model(markov_model_0_ord, markov_model_1_ord, markov_model_2_ord, char_dict):
    chars = np.arange(128)
    distribution = np.asarray(markov_model_0_ord[0], dtype=np.float)/np.sum(markov_model_0_ord[0])
    left_char = np.random.choice(chars, p=distribution)    
    distribution = np.asarray(markov_model_1_ord[0][left_char], dtype=np.float)/np.sum(markov_model_1_ord[0][left_char])
    middle_char = np.random.choice(chars, p=distribution)
    distribution = np.asarray(markov_model_2_ord[0][left_char][middle_char], dtype=np.float)/np.sum(markov_model_2_ord[0][left_char][middle_char])
    right_char = np.random.choice(chars, p=distribution)
    print(str(right_char) + ',' + str(middle_char) + ',' + str(left_char))
    left_image = char_dict[chr(left_char)]
    left_image_idx = np.random.choice(np.arange(len(left_image)))
    left_image = left_image[left_image_idx]
    right_image = char_dict[chr(right_char)]
    right_image_idx = np.random.choice(np.arange(len(right_image)))
    right_image = right_image[right_image_idx]
    center_image = char_dict[chr(middle_char)]
    center_image_idx = np.random.choice(np.arange(len(center_image)))
    center_image = center_image[center_image_idx]
    label = (chr(left_char), chr(middle_char), chr(right_char))
    
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
            axarr[i,j].set_title(str(images[x*i+j][0]))
    f.tight_layout()
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
#                   imgs.append(copy.deepcopy(img[1]))
                   imgs.append(img[1])
           char_dict[character] = imgs
         
        for asci in range(65,91):
           character = chr(asci)
           imgs = []
           for img in images: 
               if img[0]==str(character):
#                   imgs.append(copy.deepcopy(img[1]))
                   imgs.append(img[1])
           char_dict[character] = imgs
           
        for asci in range(48,58):
           character = chr(asci)
           imgs = []
           for img in images: 
               if img[0]==str(character):
#                   imgs.append(copy.deepcopy(img[1]))
                   imgs.append(img[1])
           char_dict[character] = imgs
           
        #save the dictionary
#        try:
#            with open(file_name, 'wb') as f:
#                pickle.dump(char_dict, f, pickle.HIGHEST_PROTOCOL)
#        except OSError:
#            print("to_dictionary(): Could not save dictionary")
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
  
    
def create_dataset(num_saples, overlap_frac = [0.3, 0.3], destination = ""):
    source_file_dataset_name = "emnist-byclass-train-images-idx3-ubyte"
    source_file_data_labels_name = "emnist-byclass-train-labels-idx1-ubyte"
    source_file_mappings_name = "emnist-byclass-mapping.txt"
    soure_dir = "EMNIST"
    markov_model_training_file="corncob_english.txt"
    
    print("Import dataset...")
    images = dataset_import(soure_dir + "/" + source_file_dataset_name, soure_dir + "/" + source_file_data_labels_name, 800000, soure_dir + "/" + source_file_mappings_name)
    print("Create dictionary...")
    char_dict = to_dictionary(images)
    assert(validate_dic(char_dict))
    print("Create 0th order markov model...")
    mc0o = mcl.markov_chain_of_natural_language_lite(0, markov_model_training_file)
    print("Create 1st order markov model...")
    mc1o = mcl.markov_chain_of_natural_language_lite(1, markov_model_training_file)
    print("Create 2nd order markov model...")
    mc2o = mcl.markov_chain_of_natural_language_lite(2, markov_model_training_file)
    
    overlap_images = []
    print("Concatenating images...")
    for n in range(num_saples):
        try:
            label, left_image, center_image, right_image = image_selection_markov_model(mc0o, mc1o, mc2o, char_dict)
            print(label)
            overlap_images.append([label, overlap(left_image, center_image, right_image, overlap_frac)])
        except:
            print("Character not present in the data set encounted. Skiping sample.")

        print("[Concatenating images] Status: %.2f %%" % (float(n)/float(num_saples)*100))
    print("Export overlaping characters dataset...")
    dataset_export(source_file_dataset_name + "_overlap", source_file_data_labels_name + "_overlap", overlap_images, soure_dir + "/" + source_file_mappings_name)
    print("never reached")
    
    print(len(overlap_images))
    
def validate_dic(char_dict):
    for key in char_dict.keys():
        if not char_dict[key]:
            print('fail for ' + str(key))
            return False
    return True
    
    
if __name__ == '__main__':

#    create_dataset(550000)
#    images = dataset_import("EMNIST/emnist-byclass-train-images-idx3-ubyte", "EMNIST/emnist-byclass-train-labels-idx1-ubyte", 1000, "EMNIST/emnist-byclass-mapping.txt")
    img = dataset_import("emnist-byclass-train-images-idx3-ubyte_overlap", "emnist-byclass-train-labels-idx1-ubyte_overlap", 800000, "EMNIST/emnist-byclass-mapping.txt")
    plot(img[:25])
##    char_dict = to_dictionary(images)
##    char_dict = to_dictionary()
##    path = "/home/granchgen/Dokumente/TUDelf_2nd-Q/Deep_Learning/Project/EMNIST"
##    fd = open(os.path.join(path, 'emnist-byclass-train-images-idx3-ubyte'))
##    loaded = np.fromfile(file=fd, dtype=np.uint8)
##    trainX = loaded[16:].reshape((697932, 28, 28, 1)).astype(np.float32)
#    overlap_images = []
#    file = "corncob_english.txt"
##    mc0o = mcl.markov_chain_of_natural_language_lite(0, file)
##    mc1o = mcl.markov_chain_of_natural_language_lite(1, file)
##    mc2o = mcl.markov_chain_of_natural_language_lite(2, file)
#    for n in range(9):
#        label, left_image, center_image, right_image = image_selection_markov_model(mc0o, mc1o, mc2o, char_dict)
#        overlap_images.append([label, overlap(left_image, center_image, right_image, [0.3,0.3])])
#    
#    plot(overlap_images)
#    
#    dataset_export("emnist-byclass-train-images-idx3-ubyte_overlap", "emnist-byclass-train-labels-idx1-ubyte_overlap", overlap_images, "EMNIST/emnist-byclass-mapping.txt")
#    
    
    
#    csv = np.genfromtxt ('light_emnist.csv', delimiter=",")
#    labels = csv[:,0]
#    characters = csv[:,1:]
#    car_1 = np.reshape(characters[4,:], (28,28))
#    print (labels)
#   
#    a = np.random.random((16, 16))
#    plt.imshow(car_1.T, cmap='hot', interpolation='nearest')
#    plt.show()