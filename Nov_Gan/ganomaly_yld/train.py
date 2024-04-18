"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
from lib.data_1 import load_data
from lib.model import Ganomaly
import os

##
def train():
    """ Training
    """

    ## 设置gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    ##
    # ARGUMENTS
    opt = Options().parse()
    ##
    # LOAD DATA
    dataloader = load_data(opt)
    ##
    # LOAD MODEL
    model = Ganomaly(opt, dataloader)
    ##
    # TRAIN MODEL
    model.test_3()
    #model.train()
    # result = model.test_1()
    # result = model.test_moco()
    # model.test_1()
    # print(result)

if __name__ == '__main__':
    train()
