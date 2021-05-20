from model import ResNet
import train, grad_feature,loss_feature
import torch
import torch.nn as nn
import torch.optim as optim

device = 0

#train a resnet20 model
model  = nn.DataParallel(ResNet.resnet20(10)).to(device)
optimizer = optim.SGD(model.parameters(),lr = 0.1,momentum=0.9,weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100, 150],last_epoch=-1)
standard_train = train.standard_train
pgd_train = train.pgd_train

#load pretrained model
resnet = nn.DataParallel(ResNet.resnet20(10)).to(device)
resnet.module.load_state_dict(torch.load('./pretrained/resnet20.pth'))
resnet_robust = nn.DataParallel(ResNet.resnet20(10)).to(device)
resnet_robust.module.load_state_dict(torch.load('./pretrained/resnet20_pgd.pth'))


#compare the grad features
test_dataset = train.cifar_test
train_dataset = train.cifar_train
diff_plot_grad = grad_feature.diff_plot_grad
diff_plot_loss = loss_feature.diff_plot_loss

def main():
    print('delete the hash to execute the following command')

    #standard_train(model,optimizer,scheduler,'model')
    #pgd_train(model,optimizer,scheduler,'model_pgd')
    #diff_plot_grad(resnet,resnet_robust,test_dataset,desc1='resnet',desc2='resnet_robust',savename='./image/grad_default.png')
    #diff_plot_loss(resnet,resnet_robust,test_dataset,desc1='resnet',desc2='resnet_robust',savename='./image/loss_default.png')


if __name__=='__main__':
    main()