"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data.sampler import  SubsetRandomSampler,SequentialSampler,RandomSampler
from PIL import ImageFilter
import random

def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)

    if opt.dataset in ['cifar10']:
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': False}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        classes = {
            'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }

        dataset = {}
        dataset['train'] = CIFAR10(root='./data', train=True, download=True, transform=transform)
        dataset['test'] = CIFAR10(root='./data', train=False, download=True, transform=transform)

        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_cifar_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            abn_cls_idx=classes[opt.abnormal_class],
            manualseed=opt.manualseed
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader

    elif opt.dataset in ['mnist']:
        opt.abnormal_class = int(opt.abnormal_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        dataset = {}
        dataset['train'] = MNIST(root='./data', train=True, download=True, transform=transform)
        dataset['test'] = MNIST(root='./data', train=False, download=True, transform=transform)

        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_mnist_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            abn_cls_idx=opt.abnormal_class,
            manualseed=opt.manualseed
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader

    elif opt.dataset in ['mnist2']:
        opt.abnormal_class = int(opt.abnormal_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        dataset = {}
        dataset['train'] = MNIST(root='./data', train=True, download=True, transform=transform)
        dataset['test'] = MNIST(root='./data', train=False, download=True, transform=transform)

        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_mnist2_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            nrm_cls_idx=opt.abnormal_class,
            proportion=opt.proportion,
            manualseed=opt.manualseed
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader

    else:
        normalize = transforms.Normalize((0.5,), (0.5,))
        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.Grayscale(1),
            transforms.Resize((opt.isize,opt.isize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
            
        splits = ['train', 'test']
        transform = transforms.Compose([transforms.Resize((opt.isize,opt.isize)),
                                        transforms.Grayscale(1),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),])
            
        dataset = {x: ImageFolder(os.path.join(opt.dataroot, x)) for x in splits} 
        dataset['train']=ImageFolder(os.path.join(opt.dataroot, 'train'), transform) 
        #val_size = len(dataset['train']) - 10000
        #val_dataset_1, _ = torch.utils.data.random_split(dataset['train'], [10000, val_size])
        #dataset['test']=ImageFolder(os.path.join(opt.dataroot, 'test')) 
        dataset['test']=CustomImageFolder(os.path.join(opt.dataroot, 'test'),[transform,augmentation])
        #dataset['val']=CustomImageFolder(os.path.join(opt.dataroot, 'val'),TwoCropsTransform(transforms.Compose(augmentation)))
        dataset['test0']=CustomImageFolder(os.path.join(opt.dataroot, 'test_baseline'),[transform,augmentation])
        test_size=opt.test_set_size
        #print(test_size)
        val_size = len(dataset['test']) - 2000
        val_dataset, test_dataset = torch.utils.data.random_split(dataset['test'], [2000, val_size])
        dataset['val']=val_dataset
        dataset['test']=dataset['test0']
        # dataset['test']=test_dataset
        print(len(dataset['test']))
        splits = ['train', 'test', 'val']
        drop_last_batch = {'train': True, 'test': True,'val':True}
        shuffle = {'train': False, 'test': False,'val':False}   
        #train_indices = torch.from_numpy(np.random.choice(dataset['train'],size=(opt.train_set_size,), replace=False))
        #test_indices = torch.from_numpy(np.random.choice(dataset['test'],size=(opt.test_set_size,), replace=False)) 
        #torch.manual_seed(1)
        #print(len(dataset['test']))
        #indices = list(range(len(dataset['test'])))
        #np.random.seed(3)
        #np.random.shuffle(indices)
        #random_indices = torch.randperm(opt.test_set_size)
        val_sampler=RandomSampler(dataset['val'])
        train_sampler=SequentialSampler(dataset['train'])
        #test_sampler=SubsetRandomSampler(indices=indices[:opt.test_set_size])
        test_sampler=SequentialSampler(dataset['test'])
        sampler={'train':train_sampler, 'test': test_sampler,'val':val_sampler}
        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     sampler=sampler[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
 
    
class CustomImageFolder(ImageFolder):
    """
    """
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self)) #该文件夹中的长度
        self.transform0=self.transform[0]
        self.transform1=self.transform[1]
        
    def __getitem__(self, index1):    
        path1 = self.imgs[index1][0] #此时的self.imgs等于self.samples，即内容为[(图像路径, 该图像对应的类别索引值),(),...]
        label1 = self.imgs[index1][1]

        img0 = self.loader(path1)
        if self.transform is not None:
            img1 = self.transform0(img0)
            img2 = self.transform1(img0)
            img1=[img1,img2]
        return img1, label1,path1
    
##
def get_cifar_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # Convert train-test labels into numpy array.
    trn_lbl = np.array(trn_lbl)
    tst_lbl = np.array(tst_lbl)

    # --
    # Find idx, img, lbl for abnormal and normal on org dataset.
    nrm_trn_idx = np.where(trn_lbl != abn_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl == abn_cls_idx)[0]
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.

    nrm_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # --
    if manualseed != -1:
        # Random seed.
        # Concatenate the original train and test sets.
        nrm_img = np.concatenate((nrm_trn_img, nrm_tst_img), axis=0)
        nrm_lbl = np.concatenate((nrm_trn_lbl, nrm_tst_lbl), axis=0)
        abn_img = np.concatenate((abn_trn_img, abn_tst_img), axis=0)
        abn_lbl = np.concatenate((abn_trn_lbl, abn_tst_lbl), axis=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . -> test
    #        . -> normal
    #        . -> abnormal
    new_trn_img = np.copy(nrm_trn_img)
    new_trn_lbl = np.copy(nrm_trn_lbl)
    new_tst_img = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl

##
def get_mnist_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # --
    # Find normal abnormal indexes.
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != abn_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == abn_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != abn_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == abn_cls_idx)[0])

    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # --
    if manualseed != -1:
        # Random seed.
        # Concatenate the original train and test sets.
        nrm_img = torch.cat((nrm_trn_img, nrm_tst_img), dim=0)
        nrm_lbl = torch.cat((nrm_trn_lbl, nrm_tst_lbl), dim=0)
        abn_img = torch.cat((abn_trn_img, abn_tst_img), dim=0)
        abn_lbl = torch.cat((abn_trn_lbl, abn_tst_lbl), dim=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    # Create new anomaly dataset based on the following data structure:
    new_trn_img = nrm_trn_img.clone()
    new_trn_lbl = nrm_trn_lbl.clone()
    new_tst_img = torch.cat((nrm_tst_img, abn_trn_img, abn_tst_img), dim=0)
    new_tst_lbl = torch.cat((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), dim=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl

##
def get_mnist2_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, nrm_cls_idx=0, proportion=0.5,
                               manualseed=-1):
    """ Create mnist 2 anomaly dataset.

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        nrm_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [tensor] -- New training-test images and labels.
    """
    # Seed for deterministic behavior
    if manualseed != -1:
        torch.manual_seed(manualseed)

    # --
    # Find normal abnormal indexes.
    # TODO: PyTorch v0.4 has torch.where function
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == nrm_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != nrm_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == nrm_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != nrm_cls_idx)[0])

    # Get n percent of the abnormal samples.
    abn_tst_idx = abn_tst_idx[torch.randperm(len(abn_tst_idx))]
    abn_tst_idx = abn_tst_idx[:int(len(abn_tst_idx) * proportion)]


    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    new_trn_img = nrm_trn_img.clone()
    new_trn_lbl = nrm_trn_lbl.clone()
    new_tst_img = torch.cat((nrm_tst_img, abn_tst_img), dim=0)
    new_tst_lbl = torch.cat((nrm_tst_lbl, abn_tst_lbl), dim=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl