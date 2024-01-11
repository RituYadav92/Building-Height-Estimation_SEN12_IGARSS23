import cv2 as cv
import numpy as np
import rasterio
import csv
import random
from config import *
# IMG_HEIGHT, IMG_WIDTH = 128, 128

def load_data(DATA_PATH, fname, split):
    LABELS = []
    with open(DATA_PATH/fname,'r') as f:
        for line in csv.reader(f):
            LABELS.extend(line)    
    test_size = round(len(LABELS)*split)# 0.02, 0.05
    random.seed(30)
    test_ind = random.sample(range(0, len(LABELS)), test_size)
    
    train_y, val_y = [], []
    for ind in range(0, len(LABELS)):
        if ind in test_ind:
            val_y.append(LABELS[ind])
        else:
            train_y.append(LABELS[ind])  

#     print(len(train_y), len(val_y))
    print('Load train images and masks ... ')
    
    return train_y, val_y

def scale_img(matrix):    
    max3 = int(np.max(matrix[:, :, 2]))
    min3 = int(np.min(matrix[:, :, 2]))
    # Set min/max values
    min_values = np.array([-23, -28, min3])
    max_values = np.array([0, -5, max3])

    # Reshape matrix
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float32)

    # Scale by min/max
    matrix = np.nan_to_num((matrix - min_values[None, :]) / (max_values[None, :] - min_values[None, :]))
    matrix = np.reshape(matrix, [w, h, d])

    return matrix.clip(0, 1)
    
def GRD_toRGB_S1(S1_PATH, fname):
    path_img = S1_PATH / fname

    # Read VV/VH bands
    with rasterio.open(path_img) as sar:
        sar_img = sar.read((1,2))

    sar_img = np.moveaxis(sar_img, 0, -1)
    # sar_img = imread(path_img)

    vv = sar_img[:, :, 0]
    vh = sar_img[:, :, 1]
    vv = cv.resize(vv , (IMG_HEIGHT, IMG_WIDTH), interpolation = cv.INTER_AREA)
    vh = cv.resize(vh, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv.INTER_AREA)

    x_img = np.zeros((vv.shape[0], vv.shape[1], s1_ch), dtype=np.float32)
    x_img[:, :, 0] = vv
    x_img[:, :, 1] = vh

    return scale_img(x_img)

def scale_imgS2(matrix, max_vis):
    min_values = np.array([0, 0, 0, 0, 0])
    max_values = np.array([max_vis, max_vis, max_vis, max_vis, max_vis]) # 1 in 3rd channel

    # Reshape matrix
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float32)

    # Scale by min/max
    matrix = np.nan_to_num((matrix - min_values[None, :]) / (max_values[None, :] - min_values[None, :]))
    matrix = np.reshape(matrix, [w, h, d])
    return matrix.clip(0, 1)

def GRD_toRGB_S2(S2_PATH, fname, max_vis):
    # B 4, 3, 2, 8
    path_S2 =  S2_PATH/ fname

    with rasterio.open(path_S2) as lbl:
        s2_img = lbl.read((1, 2, 3, 7, 10))#

    s2_img = np.moveaxis(s2_img, 0, -1)
    x_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, s2_ch), dtype=np.float32)
    x_img[:, :, 0] = cv.resize(s2_img[:, :, 0], (IMG_HEIGHT, IMG_WIDTH), interpolation = cv.INTER_AREA)
    x_img[:, :, 1] = cv.resize(s2_img[:, :, 1], (IMG_HEIGHT, IMG_WIDTH), interpolation = cv.INTER_AREA)
    x_img[:, :, 2] = cv.resize(s2_img[:, :, 2], (IMG_HEIGHT, IMG_WIDTH), interpolation = cv.INTER_AREA)
    x_img[:, :, 3] = cv.resize(s2_img[:, :, 3], (IMG_HEIGHT, IMG_WIDTH), interpolation = cv.INTER_AREA)
    x_img[:, :, 4] = cv.resize(s2_img[:, :, 4], (IMG_HEIGHT, IMG_WIDTH), interpolation = cv.INTER_AREA)

    x_img =  scale_imgS2(x_img, max_vis)
    return x_img