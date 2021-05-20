import torch
import torch.nn as nn
from torchvision import datasets,transforms 
from torch.utils.data import DataLoader
device = 0


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
cifar_train = datasets.CIFAR10('./data',train=True,download=True,transform=transforms.transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
cifar_test = datasets.CIFAR10('./data',train=False,download=True,transform=transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
test_loader = DataLoader(cifar_test,batch_size=512,shuffle = False,num_workers=4)
train_loader = DataLoader(cifar_train,batch_size=512,shuffle = True,num_workers=4)


#pgd and fgsm attacks
def pgd(model,data,target,epsilon = 8/255,alpha = 2* 8/255/20,step = 20):
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

#standard/adversarial train/evaluate
def epoch(model,loader,attack=None,optimizer = None,epsilon = 8/255,alpha = 2* 8/255/20,step = 20):     
    model.eval()
    if optimizer:
        model.train()
    num = 0
    total_loss = 0
    for index,(data,target) in enumerate(loader):
        data,target = data.to(device),target.to(device)
        if attack:
            pred = model(data+attack(model,data,target,epsilon,alpha,step))
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


def standard_train(model,opt,scheduler,description:str):
    baseline = 0.9
    for i in range(250):
        train_accuracy, train_loss = epoch(model,train_loader,optimizer=opt)
        test_accuracy,_ = epoch(model,test_loader)
        scheduler.step()
        if test_accuracy>=baseline:
            baseline += 0.01
            torch.save(model.module.state_dict(),'/pretrained/{}_epoch{}'.format(description,i+1))
            print('save epoch{}'.format(i+1))
        print("epoch:{}, train accuracy:{:.4}, train loss:{:.4}, test accuracy:{:.4} ".format(i+1,train_accuracy,train_loss,test_accuracy))


def pgd_train(model,opt,scheduler,description:str,epsilon = 8/255,alpha = 2* 8/255/20,step = 20):
    baseline = 0.6
    for i in range(120):
        train_accuracy, train_loss = epoch(model,train_loader,pgd,opt,epsilon,alpha,step)
        test_accuracy,_ = epoch(model,test_loader)
        test_adv_accuracy,_ = epoch(model,test_loader,pgd,epsilon,alpha,step = 20)
        scheduler.step()
        if i+1>=30 and test_adv_accuracy>baseline:
            baseline +=0.02
            torch.save(model.module.state_dict(),'/pretrained/{}_pgd_epoch{}'.format(description,i+1))
            print('save epoch{}'.format(i+1))
        print("epoch:{}, train accuracy:{:.4}, train loss:{:.4}, test accuracy:{:.4}, test adv accuracy:{:4}".format(i+1,train_accuracy,train_loss,test_accuracy,test_adv_accuracy))


