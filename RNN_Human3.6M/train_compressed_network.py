"""
    This file is the main to train compressed RNNs with GRU on the Human3.6M dataset
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

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from train import train
from test import test
from load_h36m import H36MDataSet
from MoCapRNN import MoCapRNNWavelet
import collections
import argparse
import copy
import tikzplotlib as tikz
import matplotlib.pyplot as plt
import numpy as np
'''
H36MDataSet taken from
https://github.com/hassanerrami/K_Enes/blob/0447b2115ae16edf8e07cae28bbc7425ffd8639c/RNN_baseline/load_h36m.py
(unpublished)
written by Moritz Wolter 
'''

PoseData = collections.namedtuple('PoseData', ['f', 'action', 'actor', 'array'])

CustomWavelet = collections.namedtuple('Wavelet', ['dec_lo', 'dec_hi',
                                                   'rec_lo', 'rec_hi', 'name'])

# all parameter values can be set by console command
parser = argparse.ArgumentParser(description='Train an RNN on the Human3.6m database')
parser.add_argument('--batch_size', type=int, default=50,
            help='number of mini-sequences per mini-batch, default 50')
parser.add_argument('--seq_length', type=int, default=100,
            help='number of character steps per mini-batch (chunk_size), default 100')
parser.add_argument('--epochs', type=int, default=500,
            help='number of epochs to train the network, default 500')
parser.add_argument('--pred_samples', type=int, default=50,
            help='number of samples to be predicted, default 50')
parser.add_argument('--n_hidden', type=int, default=1024,
            help='number of hidden units per layer, default 1024')
parser.add_argument('--lr', type=float, default=0.0001,
            help='learning rate, default 0.0001')
parser.add_argument('--clip', type=float, default=0.5,
            help='clipping rate, default 0.5')
parser.add_argument('--check', type=int, default=50,
            help='first epoch to check if network is the best, default 50')
parser.add_argument('--debug', type=bool, default=True,
            help='debug mode on or off, default True')
parser.add_argument('--runs', type=int, default=5,
            help='number of runs, default 5')
args = parser.parse_args()

# prints all parser arguments to let the user check their entered parameters
print('args: ', args)

# puts all parser values in easier manageable parameters
batch_size = args.batch_size
seq_length = args.seq_length
epochs = args.epochs
n_joints = 17
pred_samples = args.pred_samples
n_hidden = args.n_hidden
lr = args.lr
clip = args.clip
check = args.check
debug = args.debug
runs = args.runs

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

# the sequence length has to be larger than the length of samples to predict
assert pred_samples < seq_length

# for loop over all runs
for i in range(1, runs+1):
    print('##########')
    print('run: ', str(i))

    # loads the data and the validation data
    data = H36MDataSet(train=True, chunk_size=seq_length, dataset_name='h36mv2')
    val_data = H36MDataSet(train=False, chunk_size=seq_length,
                                     dataset_name='h36mv2')
        
    # creare wavelet
    init_wavelet = CustomWavelet(dec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                 dec_hi=[0, 0, -0.7071067811865476, 0.7071067811865476, 0, 0],
                                 rec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                 rec_hi=[0, 0, 0.7071067811865476, -0.7071067811865476, 0, 0],
                                 name='custom')
    
    # creates the net
    net = MoCapRNNWavelet(n_joints*3, n_hidden, init_wavelet=init_wavelet, mode='state_reset')
    
    # prints the net to let the user check the net structure
    print('net: ', net)
    
    pt = compute_parameter_total(net)
    print('parameter total', pt)
    
    # checks if CUDA is available and prefers GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu' :
        print('CUDA is not available, training on CPU.')
    else:
        print('CUDA is available, training on GPU.')
    
    # writer for tensorboard
    writer = SummaryWriter()  
    
    # optimization algorithm  
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    
    # loss criterion
    criterion = nn.MSELoss()
    
    # puts the net to GPU if available
    net.to(device)
    
    # puts mean and std in easier manageable parameters
    mean = data.mean 
    std = data.std
    print('Mean: ' + str(mean))
    print('Std: ' + str(std))
    
    # write any value in best_net that the parameter exists outside the for loop
    best_net = copy.deepcopy(net)

    train_losses, test_losses = [], []
    train_losses_normal, test_losses_normal = [], []
    train_losses_prl, test_losses_prl = [], []
    train_losses_acl, test_losses_acl = [], []
    test_losses_abs = []

    # for loop over all epochs
    for e in range(epochs):
        
        # puts the net into train mode
        net.train()
        
        # trains the net once
        loss, loss_normal, loss_prl, loss_acl, opt, net = train(net=net, data=data, mean=mean, std=std, 
                               criterion=criterion, opt=opt, writer=writer, 
                               epoch=e, debug=debug, batch_size=batch_size, 
                               lr=lr, clip=clip, device=device, 
                               pred_samples=pred_samples)
        
        train_losses_normal.append(loss_normal)
        train_losses_prl.append(loss_prl)
        train_losses_acl.append(loss_acl)
        train_losses.append(loss)

        # puts the net into evaluation mode
        net.eval()
        
        # tests the net once
        val_loss, val_loss_normal, val_loss_prl, val_loss_acl, val_loss_abs = test(net=net, data=val_data, mean=mean, std=std, 
                        criterion=criterion, writer=writer, 
                        epoch=e, debug=debug, batch_size=batch_size, 
                        device=device, pred_samples=pred_samples)
        
        test_losses_normal.append(val_loss_normal)
        test_losses_prl.append(val_loss_prl)
        test_losses_acl.append(val_loss_acl)
        test_losses.append(val_loss)
        test_losses_abs.append(val_loss_abs)
        
        # prints information for the user to the console
        print("Epoch: {}/{}...".format(e+1, epochs),
              "Loss: {:.4f}...".format(loss),
              "Val Loss: {:.4f}".format(val_loss))
        
        # check if current net is better than the global best one
        # only check after "check" epochs to fasten the loop a little bit
        if e == check:
            best_val_loss = val_loss
            best_loss = loss
            best_loss_abs = val_loss_abs
            best_net = copy.deepcopy(net)
        if e > check:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_loss = loss
                best_loss_abs = val_loss_abs
                best_net = copy.deepcopy(net)
    
    # close the writing on tensorboard
    writer.close()
    
    file_name = 'losses_compressed_' + str(i)
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.plot(test_losses_abs, label='Validation loss abs')
    plt.legend(frameon=False)
    plt.savefig(file_name)
    tikz.save(file_name + '.tex', standalone=True)
    plt.show()
    plt.clf()

    file_name = 'losses_compressed_normal_' + str(i)
    plt.plot(train_losses_normal, label='Training loss')
    plt.plot(test_losses_normal, label='Validation loss')
    plt.legend(frameon=False)
    plt.savefig(file_name)
    tikz.save(file_name + '.tex', standalone=True)
    plt.show()
    plt.clf()

    file_name = 'losses_compressed_wavelet_' + str(i)
    plt.plot(train_losses_prl, label='Training loss prl')
    plt.plot(train_losses_acl, label='Training loss acl')
    plt.plot(test_losses_prl, label='Validation loss prl')
    plt.plot(test_losses_acl, label='Validation loss acl')
    plt.legend(frameon=False)
    plt.savefig(file_name)
    tikz.save(file_name + '.tex', standalone=True)
    plt.show()
    plt.clf()
    
    print('Best loss: ', str(best_loss))    
    print('Best val loss: ', str(best_val_loss))
    print('Best abs loss: ', str(best_loss_abs))
        
    # produce some vids with the best net and the validation data set
    _, _, _, _, _ = test(net=best_net, data=val_data, mean=mean, std=std, 
                    criterion=criterion, writer=writer, 
                    debug=debug, batch_size=batch_size, 
                    device=device, pred_samples=pred_samples, 
                    write_vids=True, run=i, compression='compressed')
