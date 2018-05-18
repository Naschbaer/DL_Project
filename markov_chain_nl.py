#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 09:15:25 2018

@author: granchgen
"""
import numpy as np
import matplotlib.pyplot as plt

def markov_chain_of_natural_language(order: int, input_file: str):
    NUM_CHARS = 26+1
    dims = (NUM_CHARS,)
    total_num_chars = 0
    
    for n in range(order):
        dims += (NUM_CHARS,)
    markov_chain_matrix = np.zeros(dims, dtype=np.float)
    with open(input_file) as f:
        last_chars = []
        for line in f:
            last_chars = last_chars[-order:]
            for char in line:
                value = ord(char)
                if value >= 97 and value <= 122:
                    #Lower case character
                    idx = value-96
                elif value >= 65 and value <= 90:
                    #Upper case character
                    idx = value-64
                elif value == 32:
                    #Space character
                    idx = 0
                elif value == 10:
                    #New line character
                    idx = 0
#                    continue
                else:
                    continue
                    raise ValueError("Failed to create the markov chain. \"%c\" is not a valid character" % char)
                last_chars.append(idx)
                entry = tuple(last_chars[-(order+1):])
                if len(entry)<(order+1):
                    pad = tuple([0]*(order+1-len(entry)))
                    entry = pad + entry
                markov_chain_matrix[entry] += 1
                total_num_chars += 1
                
    markov_chain_matrix/=total_num_chars
    
    return markov_chain_matrix*100

def markov_chain_of_natural_language_lite(order: int, input_file: str):
    NUM_CHARS = 128
    total_num_chars = 0
    if order > 0:
        mcm = [None]*NUM_CHARS
    else:
        mcm = [0.0]*NUM_CHARS
    with open(input_file) as f:
        last_chars = []
        for line in f:
            last_chars = last_chars[-order:]
            for char in line:
                if len(last_chars)<order:
                    last_chars.append(ord(char))
                else:
                    this_mcm = mcm
                    depth = 1
                    if order != 0:
                        for last_cv in last_chars[-order:]:
                            if this_mcm[last_cv] is None:
                                if depth < order:
                                    this_mcm[last_cv] = [None]*NUM_CHARS
                                else:
                                    this_mcm[last_cv] = [0.0]*NUM_CHARS
                            this_mcm = this_mcm[last_cv]
                            depth += 1
                    
                    if ord(char) >= NUM_CHARS-1:
                        print(str(ord(char)) +","+ str(char))
                        continue
                    this_mcm[ord(char)] += 1   
                       
                    last_chars.append(ord(char))
                    total_num_chars += 1
                
    
    return mcm, total_num_chars

def string_to_entry(string: str):
    entry = ()
    for char in string:
        value = ord(char)
        if value >= 97 and value <= 122:
            #Lower case character
            idx = value-96
        elif value >= 65 and value <= 90:
            #Upper case character
            idx = value-64
        elif value == 32:
            #Space character
            idx = 0
        elif value == 10:
            #New line character
            continue
        else:
            continue
            raise ValueError("Failed to create the markov chain. \"%c\" is not a valid character" % char)
        entry += (idx,)
    return entry

def create_string(markov_chain_matrix: np.ndarray, length: int):
    order = len(markov_chain_matrix.shape)-1
    chars = np.arange(markov_chain_matrix.shape[0])
    output_string = ""
    
    if order>0:
#        dims = tuple(np.arange(order+1))
        output = [0]*order
        for i in range(length):
            distribution = markov_chain_matrix[tuple(output[-order:])]
            distribution = distribution/np.sum(distribution)
            next_char = np.random.choice(chars, p=distribution)
            output.append(next_char)
        
        for c in output:
            if c == 0:
                output_string += " "
            else:
                output_string += chr(c+96)
    return output_string

def create_string_light(mcm, num_entries, length, start):
    pass

if __name__ == '__main__':
    string_to_entry("test")
#    result = markov_chain_of_natural_language(4, "random.txt")
    result_2, num_entries = markov_chain_of_natural_language_lite(10, "random.txt")
#    result = markov_chain_of_natural_language(4, "corncob_lowercase_cleaned.txt")
#    string = create_string(result, 5000)
    string = create_string_light(result_2, num_entries, 50)

#    plt.imshow(result[1,:,:], cmap='Oranges', interpolation='none')
#    plt.show()
            