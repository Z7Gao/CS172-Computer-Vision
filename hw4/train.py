import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt


from net import GlobalCoarseNet, LocalFineNet
from loss import ScaleInvariantLoss
import util

# ignore stupid warning
import warnings
warnings.filterwarnings('ignore')

# training device
device = torch.device('cuda')

# hyperparameter
epochs_num = 4
rate = 1 # rate = 5 in paper; but result is bad so it is tuned by hand

# specifed in paper
batch_size = 32

# load dataset
dataloader_train, dataloader_valid, train_num, val_num = util.load_train(Path('nyu/train'), Path('nyu/val'), batch_size)

print("training number:{}, validation number:{}".format(train_num, val_num))

#########################################################
#              initializing the model                   #
#########################################################

# initialize and parallize the models
global_net = GlobalCoarseNet().to(device)
local_net = LocalFineNet().to(device)

# loss
global_loss = ScaleInvariantLoss()
local_loss = ScaleInvariantLoss()

# optimizer
global_optimizer = torch.optim.SGD([{'params': global_net.coarse6.parameters(), 'lr': 0.1*rate},
                                    {'params': global_net.coarse7.parameters(), 'lr': 0.1*rate}], 
                                    lr = 0.001*rate, momentum = 0.9, weight_decay = 0.1)

local_optimizer = torch.optim.SGD([{'params': local_net.fine2.parameters(), 'lr': 0.01*rate}], 
                                    lr = 0.001*rate, momentum = 0.9, weight_decay = 0.1)


# parallel
global_net = nn.DataParallel(global_net)
local_net = nn.DataParallel(local_net)

##########################################################
#       training global coarse net with validation       #
##########################################################

print("=============global coarse net training===============")

train_losses = []
val_losses = []

for e in range(epochs_num):    
    # train
    train_loss = 0
    global_net.train()
    for i, sample in enumerate(dataloader_train):
        rgb = sample[0].float().to(device)
        depth = sample[1].float().to(device)
        
        # forward pass
        output = global_net(rgb)
        loss = global_loss(output, depth)

        # back propagate
        global_net.zero_grad()
        loss.backward()

        # optimization
        global_optimizer.step()
        train_loss += loss.item()
        
    train_losses.append(train_loss / train_num)
    
    # validation - evaluation mode
    val_loss = 0
    global_net.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataloader_valid):
            rgb = sample[0].float().to(device)
            depth = sample[1].float().to(device)

            # forward pass
            output = global_net(rgb)
            loss = global_loss(output, depth)
            val_loss += loss.item()
            
    val_losses.append(val_loss / val_num)
    
    # save model
    torch.save(global_net, './models/global_net.pt')

    print(('epoch {}: train_loss = {}, validation loss = {}').format(e, train_loss, val_loss))



########################################################
#        training local fine net with validation       #
########################################################

print("=============local fine net training===============")

train_losses_ = []
val_losses_ = []

for e in range(epochs_num):    
    # train
    train_loss = 0
    local_net.train()
    for i, sample in enumerate(dataloader_train):
        rgb = sample[0].float().to(device)
        depth = sample[1].float().to(device)
        
        # results from global coarse network
        global_net.eval()
        with torch.no_grad():
            global_output = global_net(rgb).unsqueeze(1)

        # forward pass
        output = local_net(rgb, global_output).squeeze(1)
        loss = local_loss(output, depth)

        # back propagate
        local_net.zero_grad()
        loss.backward()

        # optimization
        local_optimizer.step()
        train_loss += loss.item()
        
    train_losses_.append(train_loss / train_num)

    # valid
    val_loss = 0
    local_net.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataloader_valid):
            rgb = sample[0].float().to(device)
            depth = sample[1].float().to(device)

            # results from global coarse network
            global_net.eval()
            with torch.no_grad():
                global_output = global_net(rgb).unsqueeze(1)

            # forward pass
            output = local_net(rgb, global_output).squeeze(1)
            loss = local_loss(output, depth)
            
            val_loss += loss.item()
    val_losses_.append(val_loss / val_num)
    
    # save model
    torch.save(local_net, './models/local_net.pt')

    print(('epoch {}: train_loss = {}, validation loss = {}').format(e, train_loss, val_loss))


######################################################
#      visulizing the training result on val set     #
######################################################

util.plot_loss(train_losses, val_losses)
util.plot_loss(train_losses_, val_losses_)

print("==========result visualized on validation set=============")
rgb = list(dataloader_valid)[0][0].float().to(device)
depth = list(dataloader_valid)[0][1].float().to(device)

# results from global coarse network
global_net.eval()
with torch.no_grad():
    global_output = global_net(rgb).unsqueeze(1)

# results from local fine network
local_net.eval()
with torch.no_grad():
    local_output = local_net(rgb, global_output)


for i in range(5):
    depth_g = global_output[i].view(74, 55)
    depth_l = local_output[i].view(74, 55)

    r = transforms.ToPILImage()((rgb[i]).cpu())
    gt = depth[i]

    figsize(12,4)
    plt.subplot(141); plt.title('RGB input'); plt.imshow(r)
    plt.subplot(142); plt.title('ground truth'); plt.imshow(gt.cpu())
    plt.subplot(143); plt.title('global coarse result'); plt.imshow(torch.exp(depth_g).cpu())
    plt.subplot(144); plt.title('local fine result'); plt.imshow(torch.exp(depth_l).cpu())

    plt.show()
