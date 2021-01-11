from pathlib import Path
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from PIL import Image
import os
import random

# NYU dataset statistics from google
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class NYUDepth(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform=None):
        super(NYUDepth, self).__init__()
        
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform
        

    def __len__(self):
        return len(os.listdir(self.rgb_dir))
    
	
    def __getitem__(self, idx):
        # read as PIL images
        im_array = np.load(self.rgb_dir/'{}.npy'.format(idx))
        im_array = np.einsum('ijk->kji', im_array)
        depth_array = np.load(self.depth_dir/'{}.npy'.format(idx))
        depth_array = np.einsum('ij->ji', depth_array)
        rgb_sample = Image.fromarray(im_array)
        depth_sample = Image.fromarray(depth_array)
        
        # transform
        if self.transform:
            rgb_sample = self.transform(rgb_sample)
            depth_sample = self.transform(depth_sample)
        
        # resize depth image
        depth_sample = transforms.Resize((74, 55))(depth_sample)
        
        # convert to torch tensor
        rgb_sample = transforms.ToTensor()(rgb_sample)
        rgb_sample =  transforms.Normalize(mean, std)(rgb_sample)
        depth_sample = transforms.ToTensor()(depth_sample).view(74, 55)

        return [rgb_sample, depth_sample]



def load_train(data_dir_train, data_dir_valid, batch_size):
	# data augmentation
	# trans_train = transforms.Compose([
	#				transforms.RandomRotation(5),
	#				transforms.RandomCrop((304, 228)),
	#				transforms.RandomHorizontalFlip(),
	#				transforms.Resize((304, 228)),
	#			])
	# trans_test = transforms.Resize((304, 228))

	training_set = NYUDepth(data_dir_train/'i', data_dir_train/'d', transform=transforms.Resize((304, 228)))
	dataloader_train = DataLoader(training_set, batch_size=batch_size, shuffle=True)

	valid_set = NYUDepth(data_dir_valid/'i', data_dir_valid/'d', transform=transforms.Resize((304, 228)))	
	dataloader_valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

	return dataloader_train, dataloader_valid, len(training_set ), len(valid_set)



def load_test(data_dir_test, batch_size=32):
	testing_set = NYUDepth(data_dir_test/'i', data_dir_test/'d', transform=transforms.Resize((304, 228)))
	dataloader_test = DataLoader(testing_set, batch_size=batch_size, shuffle=True)

	return dataloader_test, len(testing_set)



def plot_loss(train_losses, valid_losses):
	figsize(12, 4)
	plt.plot(train_losses, label='train losses'); plt.plot(valid_losses, label='valid losses')
	plt.xlabel("Iterations"); plt.ylabel("Loss"); plt.legend()

	plt.show()