import argparse
import sys
import os

from keras.models import Model, load_model
from keras.layers import Input
from keras import optimizers
import numpy as np

from model import build_encoder, build_decoder
from tqdm import tqdm

# import tensorflow as tf
# import keras.backend.tensorflow_backend as K

# gpuconfig = tf.ConfigProto()
# gpuconfig.gpu_options.allow_growth=True
# gpuconfig.gpu_options.per_process_gpu_memory_fraction=0.35

# sess = tf.Session(config=gpuconfig)
# K.set_session(sess)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Convolutional AutoEncoder and inference')
    parser.add_argument('--data_path', default='/home/wangyl/data_wyl/MAGAD/Comparison/OCSVM/dataset/traincae.npz', type=str, help='path to dataset')
    parser.add_argument('--train0_path', default='/home/wangyl/data_wyl/MAGAD/Comparison/OCSVM/dataset/svmtrain0.npz', type=str, help='path to dataset')
    parser.add_argument('--test_path', default='/home/wangyl/data_wyl/MAGAD/Comparison/OCSVM/dataset/svmtest.npz', type=str, help='path to dataset')
    parser.add_argument('--height', default=256, type=int, help='height of images')
    parser.add_argument('--width', default=256, type=int, help='width of images')
    parser.add_argument('--channel', default=3, type=int, help='channel of images')
    parser.add_argument('--num_epoch', default=50, type=int, help='the number of epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='mini batch size')
    parser.add_argument('--output_path', default='/home/wangyl/data_wyl/MAGAD/Comparison/OCSVM/output_cae', type=str, help='path to directory to output')

    args = parser.parse_args()

    return args


def load_data(data_to_path):
    """load data
    data should be compressed in npz
    """
    data = np.load(data_to_path)

    try:
        all_image = data['images']
        all_label = data['labels']
    except:
        print('Loading data should be numpy array and has "images" and "labels" keys.')
        sys.exit(1)

    # normalize input images
    all_image = (all_image - 127.0) / 127.0
    return all_image, all_label

# def traindata_gen(data, shape, batchsize=32):
 
#     X = np.zeros((batchsize, shape[0], shape[1], 3))
    
#     length = data.shape[0]
#     #从数据集中随机挑选batchsize对图像
#     while True:
#         n = np.random.randint(0, length, batchsize)
#         for i in range(batchsize):
#             index = n[i]
#             X[i] = data[index]

#         yield (X, X)

def traindata_gen(datapath, shape, batchsize=32):
 
    X = np.zeros((batchsize, shape[0], shape[1], 3))
    
    data = np.load(datapath)

    imagedata = data['images']
    length = imagedata.shape[0]
    
    #从数据集中随机挑选batchsize对图像
    while True:
        n = np.random.randint(0, length, batchsize)
        for i in range(batchsize):
            index = n[i]
            X[i] = imagedata[index] / 255.0

        yield (X, X)

def traindata_gen(datapath, shape, batchsize=32):
 
    X = np.zeros((batchsize, shape[0], shape[1], 3))
    
    data = np.load(datapath)

    imagedata = data['images']
    length = imagedata.shape[0]
    
    #从数据集中随机挑选batchsize对图像
    while True:
        n = np.random.randint(0, length, batchsize)
        for i in range(batchsize):
            index = n[i]
            X[i] = imagedata[index] / 255.0

        yield (X, X)

def flat_feature(enc_out):
    """flat feature of CAE features
    """
    enc_out_flat = []

    s1, s2, s3 = enc_out[0].shape
    s = s1 * s2 * s3
    for con in enc_out:
        enc_out_flat.append(con.reshape((s,)))

    return np.array(enc_out_flat)


def main():
    """main function"""
    args = parse_args()
    data_path = args.data_path
    train0_path = args.train0_path
    test_path = args.test_path
    height = args.height
    width = args.width
    channel = args.channel
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    output_path = args.output_path

    # load CIFAR-10 data from data directory
    train_datagen = traindata_gen(data_path, shape=(256,256), batchsize=batch_size)

    # build model and train
    input_img = Input(shape=(height, width, channel))
    encoder_output = build_encoder(input_img)
    decoder_output = build_decoder(encoder_output)
    autoencoder = Model(input_img, decoder_output)
    # 构建编码模型
    encoder = Model(inputs=input_img, outputs=encoder_output)
    # adm = optimizers.Adam(lr=3e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # autoencoder.compile(optimizer=adm, loss='mean_squared_error')
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    # 模型训练
    history = autoencoder.fit_generator(train_datagen,
                    epochs=num_epoch,
                    steps_per_epoch = 500,
                    verbose=1)
    
    autoencoder.save_weights(os.path.join(output_path, 'model_cae2.h5'))
    encoder.save_weights(os.path.join(output_path, 'model_e2.h5'))

    # encoder.load_weights(os.path.join(output_path, 'model_e.h5'))

    # 训练集
    data = np.load(train0_path)
    data_images = data['images']
    len = int(data_images.shape[0]/batch_size)
    for i in tqdm(range(len)):
        Y = np.zeros((batch_size, 256, 256, 3))
        for j in range(batch_size):
            Y[j] = data_images[i * batch_size + j] / 255.0
        enc_out = encoder.predict(Y)
        # flat features for OC-SVM input
        enc_out = flat_feature(enc_out)
        if i == 0:
            train_enc = enc_out
        else:
            train_enc = np.concatenate((train_enc, enc_out), axis=0)
        # save cae output
    print(train_enc.shape)
    np.savez(os.path.join(output_path, 'output_traincae2.npz'), ae_out=train_enc, labels=data['labels'])

    # 测试集
    data = np.load(test_path)
    data_images = data['images']
    len = int(data_images.shape[0]/batch_size)
    for i in tqdm(range(len)):
        Y = np.zeros((batch_size, 256, 256, 3))
        for j in range(batch_size):
            Y[j] = data_images[i * batch_size + j] / 255.0
        enc_out = encoder.predict(Y)
        # flat features for OC-SVM input
        enc_out = flat_feature(enc_out)
        if i == 0:
            test_enc = enc_out
        else:
            test_enc = np.concatenate((test_enc, enc_out), axis=0)
        # save cae output
    print(test_enc.shape)
    np.savez(os.path.join(output_path, 'output_testcae2.npz'), ae_out=test_enc, labels=data['labels'])

if __name__ == '__main__':
    # a, b = np.load('/home/wangyl/data_wyl/MAGAD/Comparison/OCSVM/output_cae/output_testcae.npz')
    # print(a.shape)
    # print(b.shape)
    main()
