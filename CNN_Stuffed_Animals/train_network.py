"""
    This file trains Alexnets on the Human3.6M dataset
    Copyright (C) 2020  Fabian Kahl, alias Fabian Phatrong  phatrong@web.de

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import torch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from AlexNet import WaveletAlexNet
import tikzplotlib as tikz
import numpy as np
import time
'''
WaveletAlexNet taken from
https://github.com/v0lta/waveletnet/blob/b0983ce66ae964fd23a76745a2e7da917de2299c/alexnet.py
(unpublished)
written by Moritz Wolter 
'''

#First, execute image_split.py to split images into train and test set

# all parameter values can be set by console command
parser = argparse.ArgumentParser(description='Training networks with and without wavelets.')
parser.add_argument('--epochs', type=int, default=250,
                    help='number of epochs per network, default 250')
parser.add_argument('--cc', type=int, default=200,
                    help='number of pixels for center crop, default 200')
parser.add_argument('--pretrained', type=bool, default=True,
                    help='use pretrained network, default True')
parser.add_argument('--lr', type=int, default=0.001,
                    help='learning rate, default 0.001')
parser.add_argument('--savepath', type=str, default='checkpoint.pth',
                    help='path to save trained network, default checkpoint.pth')
parser.add_argument('--runs', type=int, default=5,
                    help='number of runs, default 5')
args = parser.parse_args()

# prints all parser arguments to let the user check their entered parameters
print('args: ', args)

# puts all parser values in easier manageable parameters
data_dir = '../../MasterthesisImages/Images/' # directory of the images
save_path = args.savepath
sp_split = save_path.rsplit('.', 1) # splits the save_path at the last dot
image_section = args.cc
lr = args.lr
pretrained = args.pretrained
epochs = args.epochs
num_classes = 3 # number of classes to predict (Elephant, Platypus, Unicorn)
runs = args.runs

# checks if CUDA is available and prefers GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cpu' :
    print('CUDA is not available, training on CPU.')
else:
    print('CUDA is available, training on GPU.')

# transforms that are made on the train and test dataset, their loading and makes batches
transform = transforms.Compose([transforms.CenterCrop(image_section),
                                transforms.ToTensor()])
train_data = datasets.ImageFolder(data_dir + 'train', transform=transform)
test_data = datasets.ImageFolder(data_dir + 'test', transform=transform)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

# for compressed mode (wavelet) and non-compressed mode (normal) do stuff
for wavelet in ['wavelet', 'normal']:
    
    # for loop over all runs
    for run in range(runs):
        print('Wavelet:', wavelet,
              'Run:', run + 1, '/', runs)
        
        # loads Alexnet
        model = models.alexnet(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = False

        if wavelet == 'wavelet':
            # uses the wavelet classifier from wavelet Alexnet
            model_wavelet = WaveletAlexNet(num_classes=num_classes)
            classifier = model_wavelet.classifier
            model.classifier = classifier
        else:
            # uses the normal classifier from Alexnet
            model_normal = models.alexnet(pretrained=False, num_classes=num_classes)
            classifier = model_normal.classifier
            model.classifier = classifier

        if pretrained: # only trains classifier
            optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
        else: # trains everything
            optimizer = optim.Adam(model.parameters(), lr=lr)

        # uses cross entropy loss as loss
        criterion = nn.CrossEntropyLoss()
        
        # moves model to default device
        model.to(device)

        train_losses, test_losses = [], []
        time_run = 0
        
        # for loop over all epochs
        for e in range(epochs):
            
            # starts timekeeper
            time_start = time.time()
            
            running_loss = 0
            
            # runs over all training batches
            for images, labels in trainloader:
                
                # moves input and label tensors to the default device
                images, labels = images.to(device), labels.to(device)
        
                # zeroes accumulated gradients
                optimizer.zero_grad()
                
                log_ps = model(images)
                
                # calculates the loss between the output of the net and the target
                loss = criterion(log_ps, labels)
                
                # performs backprop
                loss.backward()
                
                # does one optimization step
                optimizer.step()
        
                # calculates loss
                running_loss += loss.item()
        
            else:
                test_loss = 0
                accuracy = 0
        
                # turns off the gradients for validation and computations
                with torch.no_grad():
                    
                    # runs over all test batches
                    for images, labels in testloader:
        
                        # moves input and label tensors to the default device
                        images, labels = images.to(device), labels.to(device)
        
                        # calculates test loss
                        log_ps = model(images)
                        
                        # calculates the loss between the output of the net and the target
                        test_loss += criterion(log_ps, labels)
        
                        # calculates accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
        
                train_losses.append(running_loss / len(trainloader))
                test_losses.append(test_loss / len(testloader))
                
                # stops timekeeper
                time_end = time.time()
                time_difference = time_end - time_start
                time_run += time_difference
        
                print('Epoch: {}/{}'.format(e + 1, epochs),
                      'Training Loss: {:.3f}'.format(running_loss / len(trainloader)),
                      'Test Loss: {:.3f}'.format(test_loss / len(testloader)),
                      'Test Accuracy: {:.3f}'.format(accuracy / len(testloader)),
                      'Time (in s):', time_difference)
        print('Time of this run (in s):', time_run)
        
        # computes the number of parameters used within this network
        def compute_parameter_total(net):
            total = 0
            for p in net.parameters():
                if p.requires_grad:
            
                    # print shape of layer
                    print(p.shape)
            
                    # sum up all parameters
                    total += np.prod(p.shape)
                    
            return total
        
        print('Parameters total:', compute_parameter_total(model))
        print('\n')

        # plots and saves training loss and validation loss
        losses = 'losses_' + wavelet + '_' + str(run + 1)
        plt.plot(train_losses, label='Training loss')
        plt.plot(test_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.savefig(losses)
        tikz.save(losses + '.tex', standalone=True)
        plt.show()
        plt.clf()

        # saves the model
        sp_mod = sp_split[0] + '_' + wavelet + '_' + str(run + 1) + '.' + sp_split[1]
        torch.save(model, sp_mod)
