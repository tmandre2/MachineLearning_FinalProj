#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 13:51:50 2018

@author: tmandre2
"""

import numpy as np
import tensorflow as tf
import keras
import pandas as pd 
import random
import sklearn as skl
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
data = pd.read_csv("data/protein_only.csv") 
# Preview the first 5 lines of the loaded data 
data.head()

feature_names=list(data)

#print(feature_names)

datamat=np.array(data)
random.shuffle(datamat)

labels=datamat[:,-1]
datamat=datamat[:,:-1]
