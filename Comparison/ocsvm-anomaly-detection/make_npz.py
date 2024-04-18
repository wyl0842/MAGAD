import glob
import os
import pickle
from tqdm import tqdm

import cv2
import numpy as np

# The directory you downloaded CIFAR-10
# You can download cifar10 data via https://www.kaggle.com/janzenliu/cifar-10-batches-py
data_dir = '/home/wangyl/Public_Dataset/Anomaly_Detection256'
save_dir = '/home/wangyl/data_wyl/MAGAD/Comparison/OCSVM/dataset'

train0_files = glob.glob(os.path.join(data_dir, 'train/0.normal/*.png'))
test0_files = glob.glob(os.path.join(data_dir, 'test/0.normal/*.png'))
test1_files = glob.glob(os.path.join(data_dir, 'test/1.abnormal/*.png'))

# label
# 0:airplane, 1:automobile, 2:bird. 3:cat, 4:deer, 5:dog, 6:frog, 7:horse, 8:ship, 9:truck
all_image = []
all_label = []

# # train 0.normal
# for file in tqdm(train0_files):
#     # print('Processing', file)
#     img = cv2.imread(file)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     all_image.append(img)
#     all_label.append(0)

# test 0.normal
for file in tqdm(test0_files):
    # print('Processing', file)
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    all_image.append(img)
    all_label.append(0)

# test 1.normal
for file in tqdm(test1_files):
    # print('Processing', file)
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    all_image.append(img)
    all_label.append(1)

all_images = np.array(all_image)
all_labels = np.array(all_label)

print('All images array shape')
print(all_images.shape)

print('All labeles array shape')
print(all_labels.shape)

np.savez(os.path.join(save_dir, 'svmtest.npz'), images=all_images, labels=all_labels)
