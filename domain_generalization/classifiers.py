import torch
import torch.nn as nn
from .grl import grad_reverse

class InstanceDomainClassifier(nn.Module):
    def __init__(self, num_domains):
        super(InstanceDomainClassifier,self).__init__()
        self.num_domains = num_domains
        self.dc_ip1 = nn.Linear(1024, 512)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(256,self.num_domains)
        

    def forward(self,x):
        x=grad_reverse(x)
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))

        return x
    
class ImageDomainClassifier(nn.Module):
    def __init__(self,dim,num_domains):
        super(ImageDomainClassifier,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.num_domains = num_domains
        self.Conv1 = nn.Conv2d(self.dim, 256, kernel_size=3, stride=4)
        self.Conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=4)
        self.Conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=4)
        self.Conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=4)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, self.num_domains)
        self.reLu=nn.ReLU(inplace=False)

    def forward(self,x):
        x=grad_reverse(x)
        x=self.reLu(self.Conv1(x))
        x=self.reLu(self.Conv2(x))
        x=self.reLu(self.Conv3(x))
        x=self.reLu(self.Conv4(x))
        x=self.flatten(x)
        x=self.reLu(self.linear1(x))
        x=torch.sigmoid(self.linear2(x))
        
        return x

class InstanceClassifierPrime(nn.Module):
    def __init__(self, num_cls):
        super(InstanceClassifierPrime,self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(1024, 512)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(256,self.num_cls)
        

    def forward(self,x):
        x=grad_reverse(x)
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))

        return x

class InstanceClassifier(nn.Module):
    def __init__(self, num_cls):
        super(InstanceClassifier,self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(1024, 512)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(256,self.num_cls)
        

    def forward(self,x):
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))

        return x