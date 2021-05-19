from model import ResNet,VGG19
import torch
import torch.nn as nn
from torchvision import datasets,transforms 
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
device = 0
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
cifar_train = datasets.CIFAR10('./data',train=True,download=False,transform=transforms.transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
cifar_test = datasets.CIFAR10('./data',train=False,download=False,transform=transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))

def compute_grad(model,start_index,end_index,dataset):
    l = []
    model.eval()
    for index in range(start_index,end_index):
        data = transforms.ToTensor()(dataset.data[index]).to(device).reshape(1,3,32,32)
        target = torch.tensor([dataset.targets[index]]).to(device)
        delta = torch.zeros_like(data,requires_grad=True)
        loss = nn.CrossEntropyLoss()(model(data+delta),target)
        loss.backward()
        grad = delta.grad.detach().reshape(3,32,32).cpu().numpy().transpose(1,2,0)
        l.append((dataset.data[index],grad))
    return l
    
def plot_topk(k,channel:np.ndarray):
    vec= channel.reshape(-1)
    z = np.zeros_like(vec)
    index = vec.argsort()[-k:,]
    z[index] =1
    img = z.reshape(32,32)
    return img

def plot_grad(model,start_index,end_index,dataset,k = 0):
    l = end_index-start_index
    plt.figure(figsize=(40,l*20))
    list = compute_grad(model,start_index,end_index,dataset)
    if k>0:
       for i in range(l):
        for j in range(4):
            plt.subplot(20,4,8*i+j+1)
            plt.axis('off')
            if j==0:
                plt.imshow(list[i][0])
            else:
                plt.imshow(plot_topk(k,list[i][0][:,:,j-1]))
            plt.subplot(20,4,8*i+j+5)
            plt.axis('off')
            if j==0:
                plt.imshow(list[i][1])
            else:
                plt.imshow(plot_topk(k,list[i][1][:,:,j-1]))
    else:
        for i in range(l):
            for j in range(4):
                plt.subplot(20,4,8*i+j+1)
                plt.axis('off')
                if j==0:
                    plt.imshow(list[i][0])
                else:
                    plt.imshow(list[i][0][:,:,j-1])
                plt.subplot(20,4,8*i+j+5)
                plt.axis('off')
                if j==0:
                    plt.imshow(list[i][1])
                else:
                    plt.imshow(list[i][1][:,:,j-1])




def get_test_loader(batch_size = 512):
    test_loader = DataLoader(cifar_test,batch_size,shuffle = False,num_workers=4)
    return test_loader
def get_train_loader(bat_size = 512):
    train_loader = DataLoader(cifar_train,batch_size=512,shuffle = True,num_workers=4)
    return train_loader


def resnet20():
    resnet20 = nn.DataParallel(ResNet.resnet20(10)).to(device)
    resnet20.module.load_state_dict(torch.load('./pretrained/resnet20.pth'))
    return resnet20
def resnet20_robust():
    resnet20_robust = nn.DataParallel(ResNet.resnet20(10)).to(device)
    resnet20_robust.module.load_state_dict(torch.load('./pretrained/resnet20_pgd.pth'))
    return resnet20_robust
def vgg19():
    vgg19 = nn.DataParallel(VGG19.VGG19()).to(device)
    vgg19.module.load_state_dict(torch.load('./pretrained/vgg19.pth'))
    return vgg19
def vgg19_robust():
    vgg19_robust = nn.DataParallel(VGG19.VGG19()).to(device)
    vgg19_robust.module.load_state_dict(torch.load('./pretrained/vgg19_pgd.pth'))
    return vgg19_robust

def pgd(model,data,target,epsilon = 8/255,alpha = 2* 8/255/20,step = 20,randomize = False):
    if randomize:
        delta = torch.rand_like(data,requires_grad = True)*epsilon
    else:
        delta = torch.zeros_like(data,requires_grad = True)
    for step in range(step):
        loss = nn.CrossEntropyLoss()(model(data+delta),target)
        loss.backward()
        delta.data = (delta+alpha*delta.grad.sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()
def fgsm(model, data, target, epsilon=8/255,alpha=None,step = None, randomize = None):
    delta = torch.zeros_like(data, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(data+ delta), target)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def adv_epoch(model,loader,attack=None,optimizer = None,epsilon = 8/255,alpha = 2* 8/255/20,step = 20,randomize = False):     
    model.eval()
    if optimizer:
        model.train()
    num = 0
    total_loss = 0
    for index,(data,target) in enumerate(loader):
        data,target = data.to(device),target.to(device)
        if attack:
            pred = model(data+attack(model,data,target,epsilon,alpha,step,randomize))
        else:
            pred = model(data)
        num+=pred.argmax(1).eq(target).sum().item()            
        loss = nn.CrossEntropyLoss()(pred,target)
        total_loss += loss.item()*data.shape[0]
        if optimizer:    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return num/len(loader.dataset), total_loss/len(loader.dataset)

