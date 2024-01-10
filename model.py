from tensorflow import keras
from tensorflow.keras import Model
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, add, Permute, Conv2D

def _tensor_shape(tensor):
    return getattr(tensor, '_keras_shape')

def squeeze_excite_block(input_tensor, shape, ratio=16):
    """ Create a channel-wise squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    init = input_tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = shape[channel_axis]
    # filters = 3
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)

    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def spatial_squeeze_excite_block(input_tensor):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
    Returns: a Keras tensor
    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input_tensor)

    x = multiply([input_tensor, se])
    return x

def channel_spatial_squeeze_excite(input_tensor, shape, ratio=16):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """
    cse = squeeze_excite_block(input_tensor, shape, ratio)
    sse = spatial_squeeze_excite_block(input_tensor)
    x = add([cse, sse])

    return x

# Source:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
# from .config import IMAGE_ORDERING
alpha = 1e-5
IMAGE_ORDERING = 'channels_last'

if IMAGE_ORDERING == 'channels_first':
    pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                     "releases/download/v0.2/" \
                     "resnet50_weights_th_dim_ordering_th_kernels_notop.h5"
elif IMAGE_ORDERING == 'channels_last':
    pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                     "releases/download/v0.2/" \
                     "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1


def one_side_pad(x):
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    if IMAGE_ORDERING == 'channels_first':
        x = Lambda(lambda x: x[:, :, :-1, :-1])(x)
    elif IMAGE_ORDERING == 'channels_last':
        x = Lambda(lambda x: x[:, :-1, :-1, :])(x)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING,
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = KL.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with
    strides=(2,2) and the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING, strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING,
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING,
                      strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = KL.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def get_resnet50_encoder1(input_height=224,  input_width=224,
                         pretrained='imagenet',
                         include_top=True, weights='imagenet',
                         input_tensor=None, input_shape=None,
                         pooling=None,
                         classes=1000, channels=3):

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(channels, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, channels))

    print("img_input shape", img_input.shape)
    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING,
               strides=(2, 2), name='conv1')(x)
    f1 = x

    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), data_format=IMAGE_ORDERING, strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    f2 = one_side_pad(x)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    f3 = x

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    f4 = x

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    f5 = x

    # x = AveragePooling2D(
    #     (7, 7), data_format=IMAGE_ORDERING, name='avg_pool')(x)
    # f6 = x

    # if pretrained == 'imagenet':
    #     weights_path = keras.utils.get_file(
    #         pretrained_url.split("/")[-1], pretrained_url)
    #     Model(img_input, x).load_weights(weights_path, by_name=True, skip_mismatch=True)

    return img_input, [f1, f2, f3, f4, f5]

def get_resnet50_encoder2(input_height=224,  input_width=224,
                         pretrained='imagenet',
                         include_top=True, weights='imagenet',
                         input_tensor=None, input_shape=None,
                         pooling=None,
                         classes=1000, channels=3):

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(channels, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, channels))

    # print("img_input shape", img_input.shape)
    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)
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

    x = conv_block(x, 3, [512, 512, 2048], stage=25, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=25, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=25, block='c')
    f5 = x

    # x = AveragePooling2D(
    #     (7, 7), data_format=IMAGE_ORDERING, name='avg_pool2')(x)
    # f6 = x

    if pretrained == 'imagenet':
        weights_path = keras.utils.get_file(
            pretrained_url.split("/")[-1], pretrained_url)
        Model(img_input, x).load_weights(weights_path, by_name=True, skip_mismatch=True)
        # Model(img_input, x).load_weights(weights_path)

    return img_input, [f1, f2, f3, f4, f5]

def Resnet50_UNet(n_classes,  in_img, in_inf, l1_skip_conn=True):    

    _, IMG_HEIGHT1, IMG_WIDTH1, IMG_CHANNELS1 = in_img.shape
    _, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = in_inf.shape

    img_input1, levels1 = get_resnet50_encoder1(input_height = IMG_HEIGHT1, input_width = IMG_WIDTH1, channels = IMG_CHANNELS1)
    [f11, f12, f13, f14, f15] = levels1
    # print("f1 shape", f11.shape, f12. shape, f13.shape, f14.shape, f15.shape)

    img_input2, levels2 = get_resnet50_encoder2(input_height = IMG_HEIGHT, input_width = IMG_WIDTH, channels = IMG_CHANNELS)    
    [f21, f22, f23, f24, f25] = levels2
    # print("f1 shape", f21.shape, f22. shape, f23.shape, f24.shape, f25.shape)
    f1 = Add()([f11, f21])
    f2 = Add()([f12, f22])
    f3 = Add()([f13, f23])
    f4 = Add()([f14, f24])
    # f5 = Add()([f15, f25])

    # selft attention
    # f1 = channel_spatial_squeeze_excite(f1, f1.shape)
    # f2 = channel_spatial_squeeze_excite(f2, f2.shape)
    # f3 = channel_spatial_squeeze_excite(f3, f3.shape)
    # f4 = channel_spatial_squeeze_excite(f4, f4.shape)    
    # f5 = channel_spatial_squeeze_excite(f5, f5.shape) 

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
    # print("first f2 concat", o.shape)
    o = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(o)
    o = Conv2D(128, (3, 3), padding='valid' , activation='relu' ,  name='DEC_conv3', data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization( name='DEC_bn3')(o)

    o = UpSampling2D((2, 2),  name='DEC_up3', data_format=IMAGE_ORDERING)(o)

    if l1_skip_conn:
      o = concatenate([o, f1], axis=MERGE_AXIS)
      o = channel_spatial_squeeze_excite(o, o.shape)

    o = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(o)
    o = Conv2D(64, (3, 3), padding='valid', activation='relu',  data_format=IMAGE_ORDERING, name="DEC_seg_feats")(o)
    o = BatchNormalization( name='DEC_bn4')(o)
    o = UpSampling2D((2, 2),  name='DEC_up4', data_format=IMAGE_ORDERING)(o)

    outputs = Conv2D(1, (1, 1), activation='relu',  name='HE_DEC_conv5') (o)
    print("outputs last shape", outputs.shape)

    model = Model([img_input1,img_input2], outputs)

    return model
