"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
from pytorch_msssim import ssim, max_ssim
from lib.networks import NetG, NetD, weights_init,Connection,Sia,Moco
from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.evaluate import evaluate,ACC
from PIL import Image,ImageFilter
import torchvision.transforms as transforms
import shutil

    
class BaseModel():
    """ Base Model for ganomaly
    """
    def __init__(self, opt, dataloader):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, self.opt.work_dir)
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, self.opt.test_dir)
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")

    ##
    def set_input(self, input:torch.Tensor):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])
                
    def set_moco_input(self, input:torch.Tensor):
        """ Set input and ground truth

        Args:/'
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0][0].size()).copy_(input[0][0])
            self.augpic.resize_(input[0][1].size()).copy_(input[0][1])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])
    ##
    def seed(self, seed_value):
        """ Seed 
        
        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_ssim', self.err_g_ssim.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes, fixed

    def save_weights_1(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, self.opt.work_dir, 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/netG_%02d.pth' % (weight_dir,epoch + 1))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/netD_%02d.pth' % (weight_dir,epoch + 1))
        
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, self.opt.work_dir, 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/netG.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/netD.pth' % (weight_dir))

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()
        epoch_iter = 0
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            # self.optimize()
            self.optimize_params()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                print(errors)
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes, fixed = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed)

        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))
        # self.visualizer.print_current_errors(self.epoch, errors)

    def test_1(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        if True:
            #if self.opt.load_weights:
            path = "./output/{}/{}/case/netG_119_28.pth".format(self.name.lower(), self.opt.dataset)
            pretrained_dict = torch.load(path)['state_dict']
            
            path_d = "./output/{}/{}/case/netD_119_28.pth".format(self.name.lower(), self.opt.dataset)
            pretrained_dict_d = torch.load(path_d)['state_dict']
            try:
                self.netg.load_state_dict(pretrained_dict)
                self.netd.load_state_dict(pretrained_dict_d)
            except IOError:
                raise IOError("weights not found")
            print('   Loaded weights.')
            
        self.optimizer_s = optim.Adam(self.netc.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_m = optim.Adam(self.netm.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
        self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long, device=self.device)
        self.netc.train()
        self.netm.train()
        for self.epoch in range(30):
            for data in tqdm(self.dataloader['val'], leave=False, total=len(self.dataloader['val'])):
                self.set_moco_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)              
                latent_i=latent_i.view(self.opt.batchsize,-1)
                latent_o=latent_o.view(self.opt.batchsize,-1)
                vec_rel=self.netc(latent_i)
                vec_fake=self.netm(latent_o)
                self.optimizer_s.zero_grad()
                self.optimizer_m.zero_grad()
                loss_contrastive = self.criterion(vec_rel, vec_fake, self.gt)
                loss_contrastive.backward()
                self.optimizer_s.step()
                self.optimizer_m.step()
            print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, 30))
        start_time = time.time()
        with torch.no_grad():
            self.netc.eval()
            self.netm.eval()
            file_name=[]
            batchsize=8
            for i, data in enumerate(self.dataloader['test'], 0):
                self.set_moco_input(data)
                for j in range(batchsize):
                    file_name.append(self.dataloader['test'].dataset[i*batchsize+j][2].split('/')[-2]+'/'+self.dataloader['test'].dataset[i*batchsize+j][2].split('/')[-1])
                self.fake, latent_i, latent_o = self.netg(self.input)
                latent_i=latent_i.view(self.opt.batchsize,-1)
                latent_o=latent_o.view(self.opt.batchsize,-1)
                vec_rel=self.netc(latent_i)
                vec_fake=self.netm(latent_o)
                error=1-F.cosine_similarity(vec_rel, vec_fake)
                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            print(i)
            result1 = np.array(self.an_scores.cpu())
            np.savetxt('sia_preds.txt',result1)
            result2 = np.array(self.gt_labels.cpu())
            np.savetxt('sia_gt_labels.txt',result2)
            fileObject = open('sia_filename.txt', 'w')     
            for ip in range(len(file_name)):
                fileObject.write(file_name[ip])
                fileObject.write('\n')
            fileObject.close() 
            acc,pre,rec=ACC(self.gt_labels, self.an_scores,file_name)
            print(acc)
            print(pre)
            print(rec)
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])
            elapsed = (time.time() - start_time)
            print("Time used:",elapsed)
            return performance
    
    def test_moco(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        if True:
            #if self.opt.load_weights:
            path = "./output/{}/{}/case/netG_119_28.pth".format(self.name.lower(), self.opt.dataset)
            pretrained_dict = torch.load(path)['state_dict']
            
            path_d = "./output/{}/{}/case/netD_119_28.pth".format(self.name.lower(), self.opt.dataset)
            pretrained_dict_d = torch.load(path_d)['state_dict']
            try:
                self.netg.load_state_dict(pretrained_dict)
                self.netd.load_state_dict(pretrained_dict_d)
            except IOError:
                raise IOError("weights not found")
            print('   Loaded weights.')
            
        self.optimizer_s = optim.Adam(self.netc.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_s1 = optim.Adam(self.netc_1.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_m = optim.Adam(self.netm.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
        self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long, device=self.device)
        self.zeros    = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.netc.train()
        self.netc_1.train()
        self.netm.train()
        for self.epoch in range(10):
            for data in tqdm(self.dataloader['val'], leave=False, total=len(self.dataloader['val'])):
                self.set_moco_input(data)
                #print(self.augpic.shape)
                self.fake, latent_i, latent_o = self.netg(self.input)
                _, latent_a, _ = self.netg(self.augpic)
                #ran_pic=self.input.filter(ImageFilter.BLUR)
                #latent_ran=self.netg(ran_pic)
                latent_i=latent_i.view(self.opt.batchsize,-1)
                latent_o=latent_o.view(self.opt.batchsize,-1)
                latent_a=latent_a.view(self.opt.batchsize,-1)
               
                vec_rel=self.netc(latent_i)
                vec_fake=self.netc_1(latent_o)
                vec_aug=self.netm(latent_a)
                '''
                self.optimizer_s.zero_grad()
                self.optimizer_s1.zero_grad()
                self.optimizer_m.zero_grad()
                #loss_contrastive = self.criterion(vec_rel, vec_fake, self.gt)+self.criterion(vec_rel, vec_aug, self.zeros)
                loss_contrastive=self.criterion(vec_rel, vec_fake, self.gt)+self.moco(vec_rel,vec_aug)              
                loss_contrastive.backward(retain_graph=True)
                self.optimizer_s.step()
                self.optimizer_s1.step()

                '''
                self.optimizer_s.zero_grad()
                self.optimizer_s1.zero_grad()
                #loss_contrastive_1 = self.criterion(vec_rel, vec_fake, self.gt)
                #with torch.autograd.set_detect_anomaly(True):
                #self.netc.zero_grad()
                loss_contrastive_2=self.criterion(vec_rel, vec_aug, self.zeros)+self.criterion(vec_rel, vec_fake, self.gt)
                #loss_contrastive_1.backward(retain_graph=True)
                #self.optimizer_s1.step()
                loss_contrastive_2.backward()
                self.optimizer_s.step()
                self.optimizer_s1.step()
                with torch.no_grad():                
                    for param_q, param_k in zip(self.netc.parameters(), self.netm.parameters()):
                        param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
                    
            print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, 30))
        start_time = time.time()
        with torch.no_grad():
            self.netc.eval()
            self.netc_1.eval()
            self.netm.eval()
            file_name=[]
            batchsize=8
            for i, data in enumerate(self.dataloader['test'], 0):
                self.set_moco_input(data)
                for j in range(batchsize):
                    file_name.append(self.dataloader['test'].dataset[i*batchsize+j][2].split('/')[-2]+'/'+self.dataloader['test'].dataset[i*batchsize+j][2].split('/')[-1])
                self.fake, latent_i, latent_o = self.netg(self.input)
                _, latent_a, _ = self.netg(self.augpic)
                #ran_pic=self.input.filter(ImageFilter.BLUR)
                #latent_ran=self.netg(ran_pic)
                latent_i=latent_i.view(self.opt.batchsize,-1)
                latent_o=latent_o.view(self.opt.batchsize,-1)
                latent_a=latent_a.view(self.opt.batchsize,-1)
               
                vec_rel=self.netc(latent_i)
                vec_fake=self.netc_1(latent_o)
                vec_aug=self.netm(latent_a)
                #error=1-F.cosine_similarity(vec_rel, vec_fake)+0.1*F.pairwise_distance(vec_rel, vec_aug, p=2)
                error=1-F.cosine_similarity(vec_rel, vec_fake)
                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            print(i)
            result1 = np.array(self.an_scores.cpu())
            np.savetxt('moco_preds.txt',result1)
            result2 = np.array(self.gt_labels.cpu())
            np.savetxt('moco_gt_labels.txt',result2)
            fileObject = open('moco_filename.txt', 'w')     
            for ip in range(len(file_name)):
                fileObject.write(file_name[ip])
                fileObject.write('\n')
            fileObject.close() 
            acc, pre, rec = ACC(self.gt_labels, self.an_scores, file_name)
            f1 = evaluate(self.gt_labels, self.an_scores, metric='f1_score')
            print(acc)
            print(pre)
            print(rec)
            print(f1)
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])
            elapsed = (time.time() - start_time)
            print("Time used:",elapsed)
            return performance
    
        
        
        
    def test_2(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        if True:
            #if self.opt.load_weights:
            path = "./output/{}/{}/case/netG_119_28.pth".format(self.name.lower(), self.opt.dataset)
            pretrained_dict = torch.load(path)['state_dict']
            
            path_d = "./output/{}/{}/case/netD_119_28.pth".format(self.name.lower(), self.opt.dataset)
            pretrained_dict_d = torch.load(path_d)['state_dict']
            try:
                self.netg.load_state_dict(pretrained_dict)
                self.netd.load_state_dict(pretrained_dict_d)
            except IOError:
                raise IOError("weights not found")
            print('   Loaded weights.')
            
        self.optimizer_s = optim.Adam(self.netc.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
        self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long, device=self.device)
        self.netc.train()
        for self.epoch in range(15):
            i=0
            for data in tqdm(self.dataloader['val'], leave=False, total=len(self.dataloader['val'])):
                i=i+1
                self.set_moco_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)
                #print(self.fake.shape)
                latent_o=latent_o.view(self.opt.batchsize,-1)
                vec_fake=self.netc(latent_o)
                self.optimizer_s.zero_grad()
                loss_contrastive = self.l_bce(vec_fake, self.gt)
                loss_contrastive.backward(retain_graph=True)
                self.optimizer_s.step()
            print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, 25))

        start_time = time.time()
        with torch.no_grad():
            self.netc.eval()
            self.netg.eval()
            file_name=[]
            batchsize=128

            for i, data in enumerate(self.dataloader['test'], 0):
                self.set_moco_input(data)
                for j in range(batchsize):
                    file_name.append(self.dataloader['test'].dataset[i*batchsize+j][2].split('/')[-2]+'/'+self.dataloader['test'].dataset[i*batchsize+j][2].split('/')[-1])
                self.fake, latent_i, latent_o = self.netg(self.input)
                #print(self.fake.shape)
                latent_o=latent_o.view(self.opt.batchsize,-1)
                vec_fake=self.netc(latent_o)
                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+vec_fake.size(0)] = vec_fake.reshape(vec_fake.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+vec_fake.size(0)] = self.gt.reshape(vec_fake.size(0))
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            print(i)
            result1 = np.array(self.an_scores.cpu())
            np.savetxt('1.txt',result1)
            result2 = np.array(self.gt_labels.cpu())
            np.savetxt('2.txt',result2)
            fileObject = open('3.txt', 'w')     
            path_old='/home/yinld/luntai/gan_data/test_baseline'
            path_new='/home/yinld/luntai/gan_data/final'
            for ip in range(len(file_name)):
                '''
                if(int(result1[ip]>0.6)-int(result2[ip])==0):
                    old_name=os.path.join(path_old,file_name[ip])
                    new_name=os.path.join(path_new,file_name[ip])
                    shutil.copyfile(old_name, new_name)  
                '''
                fileObject.write(file_name[ip])
                fileObject.write('\n')
            fileObject.close() 
            acc,pre,rec=ACC(self.gt_labels, self.an_scores,file_name)
            print(acc)
            print(pre)
            print(rec)
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])
            elapsed = (time.time() - start_time)
            print("Time used:",elapsed)
            return performance
        
    def test_3(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        if True:
            #if self.opt.load_weights:
            path = "./output/{}/{}/case/netG_119_28.pth".format(self.name.lower(), self.opt.dataset)
            pretrained_dict = torch.load(path)['state_dict']
            
            path_d = "./output/{}/{}/case/netD_119_28.pth".format(self.name.lower(), self.opt.dataset)
            pretrained_dict_d = torch.load(path_d)['state_dict']
            try:
                self.netg.load_state_dict(pretrained_dict)
                self.netd.load_state_dict(pretrained_dict_d)
            except IOError:
                raise IOError("weights not found")
            print('   Loaded weights.')
            
        self.optimizer_s = optim.RMSprop(self.netc.parameters(), lr=0.01, alpha=0.99, eps=1e-08)
        self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
        self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long, device=self.device)
        error = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        error1 = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        error_ssim = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        start_time = time.time()
        with torch.no_grad():
            # self.netg.eval()
            file_name = []
            batchsize = 8
            for i, data in enumerate(self.dataloader['test'], 0):
                self.set_moco_input(data)
                for j in range(batchsize):
                    file_name.append(self.dataloader['test'].dataset[i*batchsize+j][2].split('/')[-2]+'/'+self.dataloader['test'].dataset[i*batchsize+j][2].split('/')[-1])
                self.fake, latent_i, latent_o = self.netg(self.input)
                latent_i = latent_i.view(self.opt.batchsize,-1)
                latent_o = latent_o.view(self.opt.batchsize,-1)
                
                # 压缩向量作差
                # error = 5 * torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)
                # 原图重构图作差
                for k in range(batchsize):
                    error1[k] = torch.mean(abs(self.input[k]-self.fake[k]))
                # 作差 + 压缩向量作差
                error = 5 * torch.mean(torch.pow((latent_i-latent_o), 2), dim=1) + error1
                # 直接作差
                # error = error1
                # SSIM
                # for k in range(batchsize):
                #     error_ssim[k] = 1 - ssim(self.input[k].unsqueeze(0), self.fake[k].unsqueeze(0))
                # error = error_ssim
                # SSIM+残差
                # for k in range(batchsize):
                #     error_ssim[k] = 1 - ssim(self.input[k].unsqueeze(0), self.fake[k].unsqueeze(0))
                #     error1[k] = torch.mean(abs(self.input[k]-self.fake[k]))
                #     error[k] = 0.01 * self.opt.w_ssim * error_ssim[k] + 0.01 * self.opt.w_con * error1[k]
                #print(latent_i.shape)
            
                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
            an_score_mean = torch.mean(self.an_scores)
            # 线性归一化
            # self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            # Sigmoid归一化
            self.an_scores = 200 * (self.an_scores - an_score_mean)
            self.an_scores = torch.sigmoid(self.an_scores)
            print(i)
            result1 = np.array(self.an_scores.cpu())
            np.savetxt('an_scores.txt',result1)
            result2 = np.array(self.gt_labels.cpu())
            np.savetxt('gt_labels.txt',result2)
            fileObject = open('filename.txt', 'w')     
            for ip in range(len(file_name)):                
                fileObject.write(file_name[ip])
                fileObject.write('\n')
            fileObject.close() 
            acc, pre, rec = ACC(self.gt_labels, self.an_scores,file_name)
            print(acc)
            print(pre)
            print(rec)
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])
            elapsed = (time.time() - start_time)
            print(performance)
            print("Time used:",elapsed)
            return performance
        
    def test_res(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        if True:
            #if self.opt.load_weights:
            path = "/home/wangyl/data_wyl/Reconstruction/Weights/netG_119_28.pth"
            pretrained_dict = torch.load(path)['state_dict']
            
            path_d = "/home/wangyl/data_wyl/Reconstruction/Weights/netD_119_28.pth"
            pretrained_dict_d = torch.load(path_d)['state_dict']
            try:
                self.netg.load_state_dict(pretrained_dict)
                self.netd.load_state_dict(pretrained_dict_d)
            except IOError:
                raise IOError("weights not found")
            print('   Loaded weights.')
        
        
        img_list='/home/wangyl/data_wyl/Reconstruction/Dataset/test_baseline/1.abnormal'
        target_list='/home/wangyl/data_wyl/Reconstruction/Output/Nov_GAN/recon_imgs'        
        img_transforms =transforms.Compose([transforms.Resize(self.opt.isize),
                                        transforms.Grayscale(1),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),])
        self.optimizer_s = optim.RMSprop(self.netc.parameters(), lr=0.01, alpha=0.99, eps=1e-08)
        self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
        self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long, device=self.device)
        with torch.no_grad():
            self.netg.eval()
            ls = os.listdir(img_list)    
            for num in range(len(ls)):
                img_path=os.path.join(img_list,ls[num])
                img=Image.open(img_path)
                img=img.crop((0, 0, 256, 256))
                img_tensor=img_transforms(img)
                img_tensor.unsqueeze_(0)
                # 没有这句话会报错
                img_tensor = img_tensor.to(self.device)
                #self.set_input(img_tensor)
                fake, _, _ = self.netg(img_tensor)
                img_tensor=img_tensor.reshape(256,256)
                fake=fake.reshape(256,256)
                res=abs(fake-img_tensor)                
                img_tensor = (img_tensor+ 1) / 2.0* 255.0
                fake = (fake+ 1) / 2.0* 255.0
                res=(res+ 1) / 2.0* 255.0
                result=torch.cat([img_tensor,fake,res],1)
                vutils.save_image(result, '%s/%s.png' % (target_list, ls[num].split('.')[0]), normalize=True)
                
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_one_epoch()
            res = self.test()
            if res['AUC'] > best_auc:
                best_auc = res['AUC']
                self.save_weights(self.epoch)
            if self.epoch>self.opt.niter-7:
                self.save_weights_1(self.epoch)
            self.visualizer.print_current_performance(res, best_auc)
            self.visualizer.print_result(res, best_auc)
        print(">> Training model %s.[Done]" % self.name)

    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/{}/weights/netG.pth".format(self.name.lower(), self.opt.dataset,self.opt.work_dir)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'
            #self.dataloader['test'].dataset=self.dataloader['test'].dataset[0:400]
            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_o  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            # print("   Testing model %s." % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            #print(len(self.dataloader['test'].dataset))
            for i, data in enumerate(self.dataloader['test'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)
                latent_i=latent_i.view(self.opt.batchsize,-1)
                latent_o=latent_o.view(self.opt.batchsize,-1)
                error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)
                time_o = time.time()

                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
                #self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)
                #self.latent_o [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images :
                    dst = os.path.join(self.opt.outf, self.opt.name, self.opt.test_dir, 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    if i%25==0:
                        vutils.save_image(real, '%s/real_%03d.png' % (dst, i+1), normalize=True)
                        vutils.save_image(fake, '%s/fake_%03d.png' % (dst, i+1), normalize=True)#eps--png
            
            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            # auc, eer = roc(self.gt_labels, self.an_scores)
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            #self.visualizer.print_result(performance, performance['AUC'])
            return performance

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = 1-F.cosine_similarity(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      3*(label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss_contrastive
   
class MocoLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(MocoLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2,save_vec):
        #for i in range(8):
        one_distance = 1-F.cosine_similarity(output1[0:1,:], output2)
        i_distance=F.softmax(one_distance)
        save_vec[0]=-torch.log(i_distance[0])
        #moco_loss = torch.mean(save_vec)

        return save_vec[0]
   
##
class Ganomaly(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self): return 'Ganomaly'

    def __init__(self, opt, dataloader):
        super(Ganomaly, self).__init__(opt, dataloader)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0
        self.m=0.999
        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.netc = Sia(self.opt).to(self.device)
        self.netc_1 = Sia(self.opt).to(self.device)
        self.netm = Moco(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)
        self.netc.apply(weights_init)
        self.netc_1.apply(weights_init)
        for param_q, param_k in zip(self.netc.parameters(), self.netm.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient 
        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG_119.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG_119.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD_119.pth'))['state_dict'])
            print("\tDone.\n")

        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        #self.ssim = SSIM(data_range=255, size_average=True, channel=3)
        self.l_enc = l2_loss
        self.criterion = ContrastiveLoss()
        self.l_bce = nn.BCELoss()
        self.moco=MocoLoss()
        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.augpic = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.float32, device=self.device)
        self.zero   = torch.empty(size=(opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = torch.ones (size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    ##
    def forward_g(self):
        """ Forward propagate through netG
        """
        self.fake, self.latent_i, self.latent_o = self.netg(self.input)

    ##
    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())

    ##
    def backward_g(self):
        """ Backpropagate through netG
        """
        self.err_g_adv = self.l_adv(self.netd(self.input)[1], self.netd(self.fake)[1])
        self.err_g_con = self.l_con(self.fake, self.input)
        self.err_g_ssim = 1-ssim(self.input, self.fake)
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)
        self.err_g = self.err_g_adv * self.opt.w_adv+self.err_g_ssim * self.opt.w_ssim+self.opt.w_con*self.err_g_con
        self.err_g.backward(retain_graph=True)

    ##
    def backward_d(self):
        """ Backpropagate through netD
        """
        # Real - Fake Loss
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()

    ##
    def reinit_d(self):
        """ Re-initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('   Reloading net d')

    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()
        self.forward_d()

        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # netd
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        if self.err_d.item() < 1e-5: self.reinit_d()
