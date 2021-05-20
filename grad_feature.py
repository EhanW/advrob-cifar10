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



#compute gradient 
def compute_grad(model,dataset,index):
    model.eval()
    data = transforms.ToTensor()(dataset.data[index]).to(device).reshape(1,3,32,32)
    target = torch.tensor([dataset.targets[index]]).to(device)
    delta = torch.zeros_like(data,requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(data+delta),target)
    loss.backward()
    grad = delta.grad.detach().reshape(3,32,32).cpu().numpy().transpose(1,2,0)
    return grad


#compare the grad images of different models, randomly chosing num_data images in dataset
def diff_plot_grad(model1,model2,dataset,num_data=10,description1='Model1',description2='Model2',savename='./image/default.png'):
    model1.eval()
    model2.eval()
    fig = plt.figure(figsize=(10*4,10*3*num_data))  
    for i in range(num_data):
        index = np.random.randint(0,len(dataset.data))
        image = dataset.data[index]
        grad1 = compute_grad(model1,dataset,index)
        grad2 = compute_grad(model2,dataset,index)
        for j in range(4):
            if j ==0 :
                ax1 = fig.add_subplot(3*num_data,4,12*i+j+1)
                ax1.axis('off')
                ax1.set_title('Origin image, Image index:{} '.format(index))
                ax1.imshow(image)
                ax2 = fig.add_subplot(3*num_data,4,12*i+j+5)
                ax2.axis('off')
                ax2.set_title('Gradient of {}, Image index:{} '.format(description1,index))

                ax2.imshow(grad1)
                ax3 = fig.add_subplot(3*num_data,4,12*i+j+9)
                ax3.set_title('Gradient of {}, Image index:{} '.format(description2,index))
                ax3.axis('off')
                ax3.imshow(grad2)
            else:
                ax1 = fig.add_subplot(3*num_data,4,12*i+j+1)
                ax1.axis('off')
                ax1.set_title('Origin image, Image index:{}, Channel:{}'.format(index,j-1))
                ax1.imshow(image[:,:,j-1])
                ax2 = fig.add_subplot(3*num_data,4,12*i+j+5)
                ax2.axis('off')
                ax2.set_title('Gradient of {}, Image index:{}, Channel:{} '.format(description1,index,j-1))
                ax2.imshow(grad1[:,:,j-1])
                ax3 = fig.add_subplot(3*num_data,4,12*i+j+9)
                ax3.axis('off')
                ax3.set_title('Gradient of {}, Image index:{}, Channel:{} '.format(description1,index,j-1))
                ax3.imshow(grad2[:,:,j-1]) 
    fig.savefig(savename,facecolor = (1,1,1))