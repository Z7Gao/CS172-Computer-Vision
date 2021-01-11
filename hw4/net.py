import torch
import torch.nn as nn
import numpy as np


class GlobalCoarseNet(nn.Module): 
    def __init__(self):
        super(GlobalCoarseNet, self).__init__()
        
        self.coarse1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4), 
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2))

        self.coarse2 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), 
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2))
        
        self.coarse3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), 
                                     nn.ReLU())
        
        self.coarse4 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), 
                                     nn.ReLU())
        
        self.coarse5 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=2), 
                                     nn.ReLU())
        
        self.coarse6 = nn.Sequential(nn.Linear(256*8*6, 4096), 
                                     nn.ReLU(),
                                     nn.Dropout())
        
        self.coarse7 = nn.Sequential(nn.Linear(4096*1*1,74*55))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0)


    def forward(self, input):
        output = self.coarse1(input)
        output = self.coarse2(output)
        output = self.coarse3(output)
        output = self.coarse4(output)
        output = self.coarse5(output)
        output = output.reshape(output.size(0), -1)
        output = self.coarse6(output)
        output = self.coarse7(output)
        output = output.reshape(output.size(0), 74, 55)
        return output



class LocalFineNet(nn.Module):
    
    def __init__(self, init=True):
        super(LocalFineNet, self).__init__()
        
        self.fine1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=63, kernel_size=9, stride=2),
                                   #nn.BatchNorm2d(63),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        
        self.fine2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2), 
                                   #nn.BatchNorm2d(64),
                                   nn.ReLU())
        
        self.fine3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding=2)
        


class LocalFineNet(nn.Module):
    def __init__(self):
        super(LocalFineNet, self).__init__()
        
        self.fine1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=63, kernel_size=9, stride=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        
        self.fine2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2), 
                                   nn.ReLU())
        
        self.fine3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding=2))

       # pretrain the model
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0)


    def forward(self, input, global_output_batch):
        output = self.fine1(input)
        output = torch.cat((output, global_output_batch), dim=1)
        output = self.fine2(output)
        output = self.fine3(output)
        return output
