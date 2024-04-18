# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:08:43 2020

@author: YLD
"""

import torch
import torch.nn.functional as F

input1 = torch.randn(8,128)
input2 = torch.randn(8,128)

output0 = F.cosine_similarity(input1, input2, dim=0)
output = F.cosine_similarity(input1[2:3,:], input2)
aaa=F.softmax(output)
print(output0.shape)
print(output.shape)
print(aaa.shape)