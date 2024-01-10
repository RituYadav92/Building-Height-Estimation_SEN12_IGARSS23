import tensorflow as tf
tf.version.VERSION
from tensorflow.keras.layers import Input
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from pathlib import Path

from config import *

from utils import load_data
from DG import Cust_DatasetGenerator
from model import *
from losses import *
#Resnet50_UNet, Combined_HE_model,sar_encoder1, opt_encoder1, decoder1
from utils import GRD_toRGB_S1, GRD_toRGB_S2
import rasterio
import cv2 as cv
import numpy as np
# import segmentation_models as sm
# sm.set_framework('tf.keras')
# sm.framework()

import tensorflow as tf
tf.version.VERSION

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2"

# SAR encoder
sar_input = Input(shape=(model_patch, model_patch, s1_ch))
bn_axis = -1
x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(sar_input)
x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING,
            strides=(2, 2), name='conv21')(x)
f1 = x
x = BatchNormalization(axis=bn_axis, name='bn_conv21')(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), data_format=IMAGE_ORDERING, strides=(2, 2))(x)
x = conv_block(x, 3, [64, 64, 256], stage=22, block='a', strides=(1, 1))
x = identity_block(x, 3, [64, 64, 256], stage=22, block='b')
x = identity_block(x, 3, [64, 64, 256], stage=22, block='c')
f2 = one_side_pad(x)
x = conv_block(x, 3, [128, 128, 512], stage=23, block='a')
x = identity_block(x, 3, [128, 128, 512], stage=23, block='b')
x = identity_block(x, 3, [128, 128, 512], stage=23, block='c')
x = identity_block(x, 3, [128, 128, 512], stage=23, block='d')
f3 = x
x = conv_block(x, 3, [256, 256, 1024], stage=24, block='a')
x = identity_block(x, 3, [256, 256, 1024], stage=24, block='b')
x = identity_block(x, 3, [256, 256, 1024], stage=24, block='c')
x = identity_block(x, 3, [256, 256, 1024], stage=24, block='d')
x = identity_block(x, 3, [256, 256, 1024], stage=24, block='e')
x = identity_block(x, 3, [256, 256, 1024], stage=24, block='f')
f4 = x
sar_encoder1 = keras.Model(sar_input, [f1, f2, f3, f4], name="sar_encoder1")
weights_path = keras.utils.get_file(pretrained_url.split("/")[-1], pretrained_url)
sar_encoder1.load_weights(weights_path, by_name=True, skip_mismatch=True)
print(f1.shape, f2.shape, f3.shape, f4.shape)



# optical encoder
opt_input = Input(shape=(model_patch, model_patch, s2_ch))
bn_axis = -1
x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(opt_input)
x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING,
            strides=(2, 2), name='conv21')(x)
f1 = x
x = BatchNormalization(axis=bn_axis, name='bn_conv21')(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), data_format=IMAGE_ORDERING, strides=(2, 2))(x)
x = conv_block(x, 3, [64, 64, 256], stage=22, block='a', strides=(1, 1))
x = identity_block(x, 3, [64, 64, 256], stage=22, block='b')
x = identity_block(x, 3, [64, 64, 256], stage=22, block='c')
f2 = one_side_pad(x)
x = conv_block(x, 3, [128, 128, 512], stage=23, block='a')
x = identity_block(x, 3, [128, 128, 512], stage=23, block='b')
x = identity_block(x, 3, [128, 128, 512], stage=23, block='c')
x = identity_block(x, 3, [128, 128, 512], stage=23, block='d')
f3 = x
x = conv_block(x, 3, [256, 256, 1024], stage=24, block='a')
x = identity_block(x, 3, [256, 256, 1024], stage=24, block='b')
x = identity_block(x, 3, [256, 256, 1024], stage=24, block='c')
x = identity_block(x, 3, [256, 256, 1024], stage=24, block='d')
x = identity_block(x, 3, [256, 256, 1024], stage=24, block='e')
x = identity_block(x, 3, [256, 256, 1024], stage=24, block='f')
f4 = x
opt_encoder1 = keras.Model(opt_input, [f1, f2, f3, f4], name="opt_encoder1")
weights_path = keras.utils.get_file(pretrained_url.split("/")[-1], pretrained_url)
opt_encoder1.load_weights(weights_path, by_name=True, skip_mismatch=True)
print(f1.shape, f2.shape, f3.shape, f4.shape)

"""
## Fuse E1 and E2 outputs and decode to height map
"""
ch = [64, 256, 512, 1024]
f1 = keras.Input(shape=(int(model_patch/2), int(model_patch/2), ch[0]))
f2 = keras.Input(shape=(int(model_patch/4), int(model_patch/4), ch[1]))
f3 = keras.Input(shape=(int(model_patch/8), int(model_patch/8), ch[2]))
f4 = keras.Input(shape=(int(model_patch/16), int(model_patch/16), ch[3]))
o = f4
o = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(o)
o = Conv2D(512, (3, 3), padding='valid' , activation='relu' ,  name='DEC_conv1', data_format=IMAGE_ORDERING)(o)
o = BatchNormalization( name='DEC_bn1')(o)
o = UpSampling2D((2, 2),  name='DEC_up1', data_format=IMAGE_ORDERING)(o)
o = concatenate([o, f3], axis=MERGE_AXIS)
o = channel_spatial_squeeze_excite(o, o.shape)
o = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(o)
o = Conv2D(256, (3, 3), padding='valid', activation='relu' ,  name='DEC_conv2', data_format=IMAGE_ORDERING)(o)
o = BatchNormalization( name='DEC_bn2')(o)
o = UpSampling2D((2, 2),  name='DEC_up2', data_format=IMAGE_ORDERING)(o)
o = concatenate([o, f2], axis=MERGE_AXIS)
o = channel_spatial_squeeze_excite(o, o.shape)
o = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(o)
o = Conv2D(128, (3, 3), padding='valid' , activation='relu' ,  name='DEC_conv3', data_format=IMAGE_ORDERING)(o)
o = BatchNormalization( name='DEC_bn3')(o)
o = UpSampling2D((2, 2),  name='DEC_up3', data_format=IMAGE_ORDERING)(o)
o = concatenate([o, f1], axis=MERGE_AXIS)
o = channel_spatial_squeeze_excite(o, o.shape)
o = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(o)
o = Conv2D(64, (3, 3), padding='valid', activation='relu',  data_format=IMAGE_ORDERING, name="DEC_seg_feats")(o)
o = BatchNormalization( name='DEC_bn4')(o)
o = UpSampling2D((2, 2),  name='DEC_up4', data_format=IMAGE_ORDERING)(o)
outputs = Conv2D(1, (1, 1), activation='relu',  name='HE_DEC_conv5') (o)
print("Decoder output shape", outputs.shape)
decoder1 = keras.Model([f1, f2, f3, f4], outputs, name="decoder1")

class Combined_HE_model(keras.Model):
    def __init__(self, sar_encoder1, opt_encoder1, decoder1, **kwargs):
        super(Combined_HE_model, self).__init__(**kwargs)
        self.sar_encoder1 = sar_encoder1
        self.opt_encoder1 = opt_encoder1
        self.decoder1 = decoder1
        self.alpha = 0.4
        self.beta = 0.6
        self.maxDepthVal = 176.0/1.0        
        self.mse_loss_tracker = keras.metrics.Mean(name="mse_loss")
        self.val_mse_loss_tracker = keras.metrics.Mean(name="val_mse_loss")        
        self.ss_loss_tracker = keras.metrics.Mean(name="ss_loss")
        self.val_ss_loss_tracker = keras.metrics.Mean(name="val_ss_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        self.mse = tf.keras.losses.MeanSquaredError()
        self.huber = tf.keras.losses.Huber()
        self.perc = tf.keras.losses.MeanAbsolutePercentageError()
        self.cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)        
    def struct_loss(self, target, pred):
        # Structural similarity (SSIM) index        
        ssim_loss = tf.reduce_mean( 1 - tf.image.ssim(
            target, pred, max_val=self.maxDepthVal, filter_size=5, k1=0.01 ** 2, k2=0.03 ** 2 )
                                   )        
        return ssim_loss
    
    @property
    def metrics(self):
        trackers = [
            self.mse_loss_tracker,
            self.ss_loss_tracker,
            self.total_loss_tracker,
        ]        
        return trackers    
    def train_step(self, data):        
        [s1_img, s2_img], label = data
        with tf.GradientTape() as tape:            
            [o11, o12, o13, o14] = self.sar_encoder1(s1_img, training=True)
            [o21, o22, o23, o24] = self.opt_encoder1(s2_img, training=True)            
            o1 = Add()([o11, o21])
            o2 = Add()([o12, o22])
            o3 = Add()([o13, o23])
            o4 = Add()([o14, o24])
            he_out = self.decoder1([o1, o2, o3, o4], training = True)            
            #losses
            mse_loss = self.mse(label, he_out)
            ss_loss = self.cosine_loss(label, he_out)
            total_loss = (self.alpha * mse_loss) + (self.beta * ss_loss)                
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.mse_loss_tracker.update_state(mse_loss)
        self.ss_loss_tracker.update_state(ss_loss)
        self.total_loss_tracker.update_state(total_loss)
        tracker_results = {
            "mse_loss": self.mse_loss_tracker.result(),
            "ss_loss": self.ss_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
            }
        return tracker_results

    def test_step(self, data):
        [s1_img, s2_img], label = data
        [o11, o12, o13, o14] = self.sar_encoder1(s1_img, training=False)
        [o21, o22, o23, o24] = self.opt_encoder1(s2_img, training=False)        
        o1 = Add()([o11, o21])
        o2 = Add()([o12, o22])
        o3 = Add()([o13, o23])
        o4 = Add()([o14, o24])        
        he_out = self.decoder1([o1, o2, o3, o4], training = False)        
        #losses
        mse_loss_val = self.mse(label, he_out)
        ss_loss_val = self.cosine_loss(label, he_out)
        total_loss_val = (self.alpha * mse_loss_val) + (self.beta * ss_loss_val)        
        self.val_mse_loss_tracker.update_state(mse_loss_val)
        self.val_ss_loss_tracker.update_state(ss_loss_val)
        self.val_total_loss_tracker.update_state(total_loss_val)
        return {            
            "mse_loss": self.val_mse_loss_tracker.result(),
            "ss_loss": self.val_ss_loss_tracker.result(),
            "loss": self.val_total_loss_tracker.result(),
        }
    def call(self, data):
        return Combined_HE_model(self.sar_encoder1, self.opt_encoder1, self.decoder1)


def train_fusion(n_classes, S1, S2, train_y, val_y, WEIGHT_FNAME, subclass = False):    
    lr = 0.0001
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    earlystopper = EarlyStopping(patience=20, verbose=1)
    scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001)
    if subclass:
        print('in subclass model')
        model = Combined_HE_model(sar_encoder1, opt_encoder1, decoder1)
        model.compile(optimizer)        
        model.fit(my_training_batch_generator, validation_data=my_validation_batch_generator,  epochs=50,  steps_per_epoch=int(len(train_y)/train_batchSize), validation_steps=int(len(val_y)/val_batchSize) ,callbacks=[scheduler, earlystopper])
        model.save_weights(WEIGHT_PATH/WEIGHT_FNAME)
    else:
        model = Resnet50_UNet(n_classes, S1, S2)
        model.compile(optimizer, loss = mse)        
        checkpointer = ModelCheckpoint(WEIGHT_PATH/WEIGHT_FNAME, verbose=1, save_best_only=True)
        model.fit(my_training_batch_generator, validation_data=my_validation_batch_generator,  epochs=50,  steps_per_epoch=int(len(train_y)/train_batchSize), validation_steps=int(len(val_y)/val_batchSize) ,callbacks=[scheduler, earlystopper, checkpointer])    


def evaluate_fusion(weight_file, S1, S2, val_y):
    model = Combined_HE_model(sar_encoder1, opt_encoder1, decoder1)
    model.built = True
#     model = Resnet50_UNet(n_classes, S1, S2)
    model.load_weights(WEIGHT_PATH/weight_file)
    MSE = []

    OUT_FOLDER = WEIGHT_PATH / 'Pred_Mask'
    if not os.path.exists(OUT_FOLDER): os.mkdir(OUT_FOLDER)            
    for fname in val_y[1:]:
        name_Split = str.split(fname, '_')
        print(len(name_Split), name_Split)
        tmp = 6
        S1_name = name_Split[0] + '_' + name_Split[1] + '_S1_' + name_Split[3] + '_' + str(tmp)
        S2_name = name_Split[0] + '_' + name_Split[1] + '_S2_' + name_Split[3] + '_' + str(tmp)
        print('S1_name', S1_name)
        s1img = GRD_toRGB_S1(S1_PATH, S1_name)
        s2img = GRD_toRGB_S2(S2_PATH, S2_name, S2_MAX)
        with rasterio.open(LABEL10_PATH/ fname) as lbl:
            labelimg = lbl.read(1)
            crs = lbl.crs
            transform = lbl.transform
        labelimg = cv.resize(labelimg, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv.INTER_AREA)
        labelimg = np.nan_to_num(labelimg)

        in_s1img = tf.expand_dims(s1img, axis=0)
        in_s2img = tf.expand_dims(s2img, axis=0)
        
        [o11, o12, o13, o14] = model.sar_encoder1(in_s1img)
        [o21, o22, o23, o24] = model.opt_encoder1(in_s2img)

        o1 = Add()([o11, o21])
        o2 = Add()([o12, o22])
        o3 = Add()([o13, o23])
        o4 = Add()([o14, o24])

        pred_mask = model.decoder1([o1, o2, o3, o4])
#         pred_mask = model.predict([in_s1img, in_s2img])
        pred_mask = np.squeeze(pred_mask[0])
        
        MSE.append(np.nan_to_num(mean_squared_error(labelimg, pred_mask)))        
#         filename = fname + '_predMask.tif'    
#         profile = {'driver': 'GTiff', 'crs': crs, 'transform': transform, 'height': 1280, 'width': 1280, 'count': 1, 'dtype': labelimg.dtype, }  
#         with rasterio.open(OUT_PATH/filename, 'w', **profile) as dst:
#             dst.write(pred_mask, indexes=1)
      
    AVG_MSE = sum(MSE)/len(MSE)
    print('Average MSE :', AVG_MSE)




if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train the network.')
    parser.add_argument("mode",
                        metavar="<command>",
                        help="'train' or 'evaluate'")
    parser.add_argument('--weight', required=True,
                        default='test.h5',
                        metavar="/path/to/weightfile/",
                        help='weight directory (default=logs/)')    
    args = parser.parse_args()
    
    # load data
    train_y, val_y = load_data(DATA_PATH, LABEL_fname, splits)
    my_training_batch_generator = Cust_DatasetGenerator(train_y, batch_size=train_batchSize)
    my_validation_batch_generator = Cust_DatasetGenerator(val_y, batch_size=val_batchSize)
    
    # define network parameters    
    n_classes = 1
    S1 = Input(shape=(IMG_HEIGHT, IMG_WIDTH, s1_ch))
    S2 = Input(shape=(IMG_HEIGHT, IMG_WIDTH, s2_ch))
    
    # define loss, optimizer, lr etc.
    mse = tf.keras.losses.MeanSquaredError()    
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=0.00001)

    if args.mode == "train":
        print('In training Fusion Network')
        train_fusion(n_classes, S1, S2, train_y, val_y, args.weight, subclass = True)        
        
    if args.mode == "evaluate":
        print('Evaluating Fusion Network')
        WEIGHT_FNAME = args.weight
        evaluate_fusion(WEIGHT_FNAME, S1, S2, val_y)