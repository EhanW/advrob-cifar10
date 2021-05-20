from model import ResNet,VGG19
import torch
import torch.nn as nn
from torchvision import datasets,transforms 
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.linalg as LA
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

device = 0


#load pretrained model
def resnet20():
    resnet20 = nn.DataParallel(ResNet.resnet20(10)).to(device)
    resnet20.module.load_state_dict(torch.load('./pretrained/resnet20.pth'))
    return resnet20

def resnet20_robust():
    resnet20_robust = nn.DataParallel(ResNet.resnet20(10)).to(device)
    resnet20_robust.module.load_state_dict(torch.load('./pretrained/resnet20_pgd.pth'))
    return resnet20_robust

