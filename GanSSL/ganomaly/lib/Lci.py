import torch
import torch.nn.functional as F

###########################################
# 将部分patch擦除,保留原图外框，返回图像和掩模
###########################################
def patch_erase(img, patch_sz=(96, 96)): # img:[batch_size,3,width,height]
    im_shape = img.size()
    # print(im_shape)
    pad_sz = [im_shape[2] - patch_sz[0], im_shape[3] - patch_sz[1]]  # 留下部分宽度
    patch_mask = torch.ones([im_shape[0], im_shape[1], patch_sz[0], patch_sz[1]]).cuda()   # [batchsize, 3, 16, 16]
    dim = (pad_sz[1] // 2, pad_sz[1] // 2, pad_sz[0] // 2, pad_sz[0] // 2, 0, 0, 0, 0)
    patch_mask = F.pad(patch_mask, dim, "constant", value=0)
    # 返回处理后的图和掩模
    return img * (1. - patch_mask) + 0.1 * patch_mask * torch.randn(im_shape).cuda(), 1. - patch_mask