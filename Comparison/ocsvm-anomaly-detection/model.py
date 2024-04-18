from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Convolution2D, Activation, BatchNormalization
from keras.models import Model
# from keras.layers.normalization import BatchNormalization
# from tensorflow.keras.layers import Conv2DTranspose
import keras

# def build_cae_model(height=32, width=32, channel=3):
#     """
#     build convolutional autoencoder model
#     """
#     input_img = Input(shape=(height, width, channel))

#     # encoder
#     net = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
#     net = MaxPooling2D((2, 2), padding='same')(net)
#     net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
#     net = MaxPooling2D((2, 2), padding='same')(net)
#     net = Conv2D(4, (3, 3), activation='relu', padding='same')(net)
#     encoded = MaxPooling2D((2, 2), padding='same', name='enc')(net)

#     # decoder
#     net = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
#     net = UpSampling2D((2, 2))(net)
#     net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
#     net = UpSampling2D((2, 2))(net)
#     net = Conv2D(16, (3, 3), activation='relu', padding='same')(net)
#     net = UpSampling2D((2, 2))(net)
#     decoded = Conv2D(channel, (3, 3), activation='sigmoid', padding='same')(net)

#     return Model(input_img, decoded)

# def build_cae_model(height=32, width=32, channel=3):
#     """
#     build convolutional autoencoder model
#     """
#     input_img = Input(shape=(height, width, channel))

#     net = Conv2D(32, (3, 3), strides=(2, 2), padding='same',\
#             kernel_regularizer=keras.regularizers.l2(0.001))(input_img)   
#     BatchNormalization()
#     net = Conv2D(16, (3, 3), strides=(2, 2), padding='same')(net)  
#     BatchNormalization()
#     net = Conv2D(8, (3, 3), strides=(2, 2), padding='same')(net)  
#     BatchNormalization()
#     net = Conv2D(2, (3, 3), strides=(2, 2), padding='same')(net)  
#     BatchNormalization(name='enc')

#     net = Conv2DTranspose(2, (3, 3), strides=(2, 2), padding='same')(net)  
#     BatchNormalization()
#     net = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same')(net)  
#     BatchNormalization()
#     net = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(net)  
#     BatchNormalization()
#     net = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(net) 
#     BatchNormalization()
#     decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(net)  
#     return Model(input_img, decoded)

# def build_encoder(input_img):
#     """
#     build convolutional autoencoder model
#     """

#     net = Conv2D(32, (3, 3), strides=(2, 2), padding='same',\
#             kernel_regularizer=keras.regularizers.l2(0.001))(input_img)   
#     BatchNormalization()
#     net = Conv2D(16, (3, 3), strides=(2, 2), padding='same')(net)  
#     BatchNormalization()
#     net = Conv2D(8, (3, 3), strides=(2, 2), padding='same')(net)  
#     BatchNormalization()
#     net = Conv2D(2, (3, 3), strides=(2, 2), padding='same')(net)  
#     BatchNormalization()
#     return net

# def build_decoder(net):
#     net = Conv2DTranspose(2, (3, 3), strides=(2, 2), padding='same')(net)  
#     BatchNormalization()
#     net = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same')(net)  
#     BatchNormalization()
#     net = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(net)  
#     BatchNormalization()
#     net = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(net) 
#     BatchNormalization()
#     decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(net)  
#     return decoded

def build_encoder(input_img):
    x = Convolution2D(8, (3,3),padding='same')(input_img)
    x = Convolution2D(16, (2,2),padding='same',dilation_rate=2)(x)
    x = Convolution2D(32, (3,3),padding='same')(x)
    x=  BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Convolution2D(64, (2,2),padding='same',dilation_rate=2)(x)
    x = Convolution2D(64, (3,3),padding='same',strides=(2,2))(x)
    x=  BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Convolution2D(128, (2,2),padding='same',dilation_rate=2)(x)
    x = Convolution2D(128, (3, 3),strides=(2,2),padding='same')(x)
    x=  BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Convolution2D(256, (3,3),strides=(2,2),padding='same')(x)
    x=  BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(64, (1,1),padding='same')(x)
    x=  BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def build_decoder(x):
    x = Convolution2D(256, (2, 2),padding='same')(x)
    x=  BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)

    x = Convolution2D(128, (3, 3),padding='same')(x)
    x=  BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)

    x = Convolution2D(64, (2, 2),padding='same')(x)
    x=  BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)

    x = Convolution2D(32, (3, 3),padding='same')(x)
    x=  BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)

    x = Convolution2D(16, (2, 2),dilation_rate=2,activation='relu',padding='same')(x)
    x = Convolution2D(16, (3, 3),padding='same')(x)
    x=  BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)

    x = Convolution2D(8, (2, 2),dilation_rate=2,activation='relu',padding='same')(x)
    x = Convolution2D(8, (3, 3),padding='same')(x)
    x=  BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Convolution2D(3, (1, 1), activation='sigmoid', padding='same')(x)
    return decoded
