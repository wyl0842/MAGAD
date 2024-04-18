# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 14:08:44 2020

@author: YLD
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from lib.memory_module import MemModule
##
def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

###
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf
        last_size=4
        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        #原始大小为4
        while csize >last_size:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4 ，16*16
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, last_size, 1, 0, bias=False))
            
        if add_final_conv==False:
            main.add_module('final-{0}-{1}-conv-set'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, 1, 1, 0, bias=False))
        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

##

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Connection(nn.Module):
    def __init__(self, nz,ngpu):
        super(Connection, self).__init__()
        self.ngpu = ngpu
        main = nn.Sequential()
        '''
        main.add_module('pyramid-layers-{0}-{1}-fcl'.format(4096, 1024),
                            nn.Linear(4096, 1024))
        main.add_module('pyramid-{0}-batchnorm'.format(1024),
                            nn.BatchNorm1d(1024))
        main.add_module('pyramid-{0}-relu'.format(1024),
                        nn.ReLU(True))
        '''
        main.add_module('pyramid-layers-{0}-{1}-fcl'.format(4096, 1024),
                            nn.Linear(4096, 1024))
        main.add_module('pyramid-{0}-batchnorm'.format(1024),
                            nn.BatchNorm1d(1024))
        main.add_module('pyramid-{0}-relu'.format(1024),
                        nn.ReLU(True))
        main.add_module('pyramid-layers-{0}-{1}-fcl'.format(nz, 512),
                            nn.Linear(nz, 512))
        main.add_module('pyramid-{0}-batchnorm'.format(512),
                            nn.BatchNorm1d(512))
        main.add_module('pyramid-{0}-relu'.format(512),
                        nn.ReLU(True))
        main.add_module('pyramid-layers-{0}-{1}-fcl'.format(512, 256),
                            nn.Linear(512, 256))
        main.add_module('pyramid-{0}-batchnorm'.format(256),
                            nn.BatchNorm1d(256))
        main.add_module('pyramid-{0}-relu'.format(256),
                        nn.ReLU(True))
        main.add_module('pyramid-layers-{0}-{1}-fcl'.format(256, 128),
                            nn.Linear(256, 128))
        main.add_module('pyramid-{0}-batchnorm'.format(128),
                            nn.BatchNorm1d(128))
        main.add_module('pyramid-{0}-sigmoid'.format(128),
                       nn.ReLU(True))

        main.add_module('pyramid-layers-{0}-{1}-fcl'.format(128, 1),
                            nn.Linear(128, 1))
        main.add_module('pyramid-{0}-batchnorm'.format(1),
                            nn.BatchNorm1d(1))
        main.add_module('pyramid-{0}-sigmoid'.format(1),
                        nn.Sigmoid())

        self.main = main
        
    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output
'''        
class Allc(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, opt):
        super(Allc, self).__init__()
        self.model_allc = Connection(opt.ngpu,opt.nz)

    def forward(self, x):
        features = self.model_allc(x)
        return  features
'''   
class Sia(nn.Module):
    def __init__(self, opt):
        super(Sia, self).__init__()
        self.model_1=  Connection(opt.nz,opt.ngpu)
        self.batchsize=opt.batchsize
        #self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        x=x.view(self.batchsize, -1)
        temp = self.model_1(x)
        return temp    
##
class NetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, opt):
        super(NetD, self).__init__()
        model = Encoder(opt.isize, 1, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = Sia(opt)
        
    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features

##
