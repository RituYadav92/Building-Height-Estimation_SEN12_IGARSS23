import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import rasterio
import random
from utils import GRD_toRGB_S1, GRD_toRGB_S2
from pathlib import Path
from config import *

class Cust_DatasetGenerator2(tf.keras.utils.Sequence):
    def __init__(self, label_files, batch_size = 64):              
        # self.s1_files = s1_files
        # self.s2_files = s2_files        
        self.label_files = label_files       
        self.batch_size = batch_size
        self.n = len(self.label_files)
        self.on_epoch_end()        

    def __len__(self):
        return self.n // self.batch_size

    def __getitem__(self, idx):

        while True:
            batch_ind = np.random.choice(len(self.label_files), self.batch_size)          
            batch_input_s1img1  = []
            batch_input_s2img1 = []
            batch_input_s1img2  = []
            batch_input_s2img2 = []
            batch_input_labelimg  = []
            batch_output = []

            for ind in batch_ind:
                name_Split = str.split(self.label_files[ind], '_')
                tmp1 = random.randint(0, 5)
                S1_name1 = name_Split[0] + '_' + name_Split[1] + '_S1_' + name_Split[3] + '_' + str(tmp1)
                S2_name1 = name_Split[0] + '_' + name_Split[1] + '_S2_' + name_Split[3] + '_' + str(tmp1)
                s1img1 = GRD_toRGB_S1(S1_PATH, S1_name1)                
                s2img1 = GRD_toRGB_S2(S2_PATH, S2_name1, S2_MAX)
                
                tmp2 = random.randint(0, 5)
                S1_name2 = name_Split[0] + '_' + name_Split[1] + '_S1_' + name_Split[3] + '_' + str(tmp2)
                S2_name2 = name_Split[0] + '_' + name_Split[1] + '_S2_' + name_Split[3] + '_' + str(tmp2)
                s1img2 = GRD_toRGB_S1(S1_PATH, S1_name2)                
                s2img2 = GRD_toRGB_S2(S2_PATH, S2_name2, S2_MAX)

                with rasterio.open(LABEL10_PATH/ self.label_files[ind]) as lbl:
                    labelimg = lbl.read(1)

                labelimg = cv.resize(labelimg, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv.INTER_AREA)
                labelimg = np.nan_to_num(labelimg)
 
                batch_input_s1img1 += [ s1img1]
                batch_input_s2img1 += [ s2img1]
                
                batch_input_s1img2 += [ s1img2]
                batch_input_s2img2 += [ s2img2]

                labelimg = labelimg.astype(np.float32)            
                labelimg = labelimg.reshape((labelimg.shape[0],labelimg.shape[1], 1))
                batch_input_labelimg += [ labelimg]

            batch1_s1 = np.array( batch_input_s1img1 )
            batch1_s2 = np.array( batch_input_s2img1 )
            batch2_s1 = np.array( batch_input_s1img2 )
            batch2_s2 = np.array( batch_input_s2img2 )
            batch_label = np.array( batch_input_labelimg )

            return ([batch1_s1, batch1_s2, batch2_s1, batch2_s2], batch_label)
        
class Cust_DatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, label_files, batch_size = 64):  
  
        self.label_files = label_files       
        self.batch_size = batch_size
        self.n = len(self.label_files)
        self.on_epoch_end()        

    def __len__(self):
        return self.n // self.batch_size

    def __getitem__(self, idx):

        while True:
            batch_ind = np.random.choice(len(self.label_files), self.batch_size)          
            batch_input_s1img  = []
            batch_input_s2img = []
            batch_input_labelimg  = []
            batch_output = []

            for ind in batch_ind:
                name_Split = str.split(self.label_files[ind], '_')
                tmp = random.randint(0, 5)

                S1_name = name_Split[0] + '_' + name_Split[1] + '_S1_' + name_Split[3] + '_' + str(tmp)
                S2_name = name_Split[0] + '_' + name_Split[1] + '_S2_' + name_Split[3] + '_' + str(tmp)

                s1img = GRD_toRGB_S1(S1_PATH, S1_name)
                # s2img = s1img
                s2img = GRD_toRGB_S2(S2_PATH, S2_name, 3000)

                with rasterio.open(LABEL10_PATH/ self.label_files[ind]) as lbl:
                    labelimg = lbl.read(1)
                labelimg = cv.resize(labelimg, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv.INTER_AREA)

                labelimg = np.nan_to_num(labelimg)

                batch_input_s1img += [ s1img]
                batch_input_s2img += [ s2img]

                labelimg = labelimg.astype(np.float32)            
                labelimg = labelimg.reshape((labelimg.shape[0],labelimg.shape[1], 1))
                batch_input_labelimg += [ labelimg]

            batch_s1 = np.array( batch_input_s1img )
            batch_s2 = np.array( batch_input_s2img )
            batch_label = np.array( batch_input_labelimg )

            return ([batch_s1, batch_s2], batch_label)