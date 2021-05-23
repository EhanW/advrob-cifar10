
import torch
import torch.nn as nn
from torchvision import transforms 
import torch.linalg as LA
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
#from matplotlib.ticker import LinearLocator
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.ticker import LinearLocator, FormatStrFormatter



device = 0
#compute the gradient dirction at an image with a given index in a dataset
#return the sign(gradient) since we use l0 direction in  FGSM/PGD attack
def get_grad_direction(model,dataset,index):
    model.eval()
    img = dataset.data[index]
    lab = dataset.targets[index]
    data = transforms.ToTensor()(img).reshape(1,3,32,32).to(device)
    target = torch.tensor([lab]).to(device)
    delta = torch.zeros_like(data,requires_grad=True)
    pred = model(data+delta)
    loss = nn.CrossEntropyLoss()(pred,target)
    loss.backward()
    grad = delta.grad.detach()
    #grad_direction = grad/LA.norm(grad)
    grad_direction = grad.sign()
    return grad_direction
#compute loss along a vector
def loss_along_vector(model,dataset,index,vector):
    model.eval()
    img = dataset.data[index]
    lab = dataset.targets[index]
    data = transforms.ToTensor()(img).reshape(1,3,32,32).to(device)
    target = torch.tensor([lab]).to(device)
    loss= nn.CrossEntropyLoss()(model(data+vector),target).item()
    return loss

#compare model1&model2's loss surface over a plane spanned by two given directions around an image
def diff_plot_loss(model1,model2,dataset,num_data=10,num_direction = 4,scope = 0.6,interval = 0.02,desc1='Model 1',desc2='Model 2',savename='./image/default.png'):
    fig = plt.figure(figsize=(10*num_direction,10*2*num_data))
    X = np.arange(-scope, scope, interval)
    Y = np.arange(-scope, scope, interval)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    for i in range(num_data):
        index = np.random.randint(0,len(dataset.data))            
        direction1 = get_grad_direction(model1,dataset,index)
        direction2 = get_grad_direction(model2,dataset,index)
        for j in range(num_direction):
            rand_direction = torch.randn_like(direction1).sign()
            #plot ax1
            ax1 = fig.add_subplot(2*num_data,num_direction,2*num_direction*i+j+1,projection='3d')
            ax1.set_title('{}, Image index:{}, Random axis number:{}'.format(desc1,index,j+1))
            for r in range(Z.shape[0]):
                for c in range(Z.shape[1]):
                    x = X[r,c]
                    y = Y[r,c]
                    z = loss_along_vector(model1,dataset,index,x*direction1+y*rand_direction)
                    Z[r,c] =z
            ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm)

            #plot ax2
            ax2 = fig.add_subplot(2*num_data,num_direction,2*num_direction*i+j+1+num_direction,projection='3d')
            ax2.set_title('{}, Image index:{}, Random axis number:{}'.format(desc2,index,j+1))

            for r in range(Z.shape[0]):
                for c in range(Z.shape[1]):
                    x = X[r,c]
                    y = Y[r,c]
                    z = loss_along_vector(model2,dataset,index,x*direction2+y*rand_direction)
                    Z[r,c] =z
            ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    fig.savefig(savename,facecolor=(1,1,1),bbox_inches = 'tight')

