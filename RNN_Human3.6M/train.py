"""
    This file trains a network one epoch
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

import numpy as np
import torch
from torch import nn
import util_pytorch as up

def train(net, data, mean, std, criterion, opt, writer, epoch, debug=False, batch_size=50, lr=0.001, clip=1, device='cpu', pred_samples=50):
    '''
    Arguments
    ---------
    net: MoCapRNN network
    data: Human3.6M data to train the network
    mean: mean
    std: standard deviation
    criterion: loss criterion
    opt: optimizer
    writer: tensorboard writer
    epoch: current epoch
    debug: debug mode on or off
    batch_size: number of mini-sequences per mini-batch
    lr: learning rate
    clip: gradient clipping
    device: cuda or cpu
    pred_samples: number of samples to be predicted
    '''
    
    '''
    based on def train() from
    https://github.com/udacity/deep-learning-v2-pytorch/blob/3a95d118f9df5a86826e1791c5c100817f0fd924/recurrent-neural-networks/char-rnn/Character_Level_RNN_Solution.ipynb
    '''

    # initializes hidden state
    h = net.init_hidden(batch_size, device)
    
    # loss
    losses = []
    losses_normal = []
    losses_prl = []
    losses_acl = []
    
    # calculates all inputs and targets batches
    batch_lst = up.organize_into_batches(data.get_batches(), batch_size)
    all_x, all_y = up.get_input_target_batches(batch_lst, pred_samples)
    
    # for loop over all batches
    for i in range(len(all_x)):

        # picks the ith input and target batch
        x = all_x[i]
        y = all_y[i]
        
        # converts the 3d coords of the last two dims into one dim
        x = np.reshape(x, (len(x),len(x[0]),-1))
        y = np.reshape(y, (len(y),len(y[0]),-1))

        # normalizes the input and target
        x = (x - data.mean) / data.std
        y = (y - data.mean) / data.std

        # ensures that input and target have the same data type
        inputs, targets = torch.from_numpy(x.astype('float32')), torch.from_numpy(y.astype('float32'))

        # puts all parameters on the GPU if possible
        inputs, targets, h = inputs.to(device), targets.to(device), h.to(device)
        
        # is needed, otherwise would backprop through
        # the entire training history of all batches of this epoch
        h = h.data

        # zeroes accumulated gradients
        net.zero_grad()

        # gets the output and the hidden state from the net
        max_time = inputs.shape[1]
        for t in range(0, max_time):
            output, h = net(inputs[:,t,:], h)
        
        predict = []
        max_time = targets.shape[1]
        for t in range(0, max_time):
            output, h = net(output, h)
            predict.append(output)
        output = torch.stack(predict, 1)

        # calculates the loss between the output of the net and the target
        loss_normal = criterion(output, targets)
        prl, acl = net.get_wavelet_loss()
        if prl == -1 and acl == -1:
            # -1 means this loss does not exist
            loss = loss_normal
            losses_prl.append(0.)
            losses_acl.append(0.)
        else:
            loss = loss_normal + prl + acl
            losses_prl.append(prl.item())
            losses_acl.append(acl.item())

        # collects the loss
        losses_normal.append(loss_normal.item() * std + mean)
        losses.append(loss.item() * std + mean)

    	# performs backprop
        loss.backward()
        
        if debug == True:
            # writes gradients before clipping to tensorboard
            writer.add_histogram('gradients/before_clipping', net.W_proj.weight.grad, epoch)
        
        # helps to prevent the exploding gradient problem
        nn.utils.clip_grad_value_(net.parameters(), clip)
        if debug == True:
            # writes gradients after clipping to tensorboard
            writer.add_histogram('gradients/after_clipping', net.W_proj.weight.grad, epoch)

    	# does one optimization step
        opt.step()
        
# =============================================================================
#         if debug == True:
#             # write joint 0 to tensorboard
#             inputs_plt = inputs.to('cpu').numpy()
#             inputs_plt = np.reshape(inputs_plt, [inputs_plt.shape[0], inputs_plt.shape[1], 17, 3])
#             output_plt = output.to('cpu').detach().numpy()
#             output_plt = np.reshape(output_plt, [output_plt.shape[0], output_plt.shape[1], 17, 3])
#             targets_plt = targets.to('cpu').numpy()
#             targets_plt = np.reshape(targets_plt, [targets_plt.shape[0], targets_plt.shape[1], 17, 3])
#             full_inputs_plt = np.concatenate((inputs_plt, targets_plt), 1)
#             full_output_plt = np.concatenate((inputs_plt, output_plt), 1)
#             plot = plt.figure()
#             plt.plot(full_inputs_plt[0,:,0])
#             plt.legend(['x','y','z'])
#             writer.add_figure('joint_0_train/before_learning', plot, epoch)
#             plt.close()
#             plot = plt.figure()
#             plt.plot(full_output_plt[0,:,0])
#             plt.legend(['x','y','z'])
#             writer.add_figure('joint_0_train/after_learning', plot, epoch)
#             plt.close()
# =============================================================================
        
    # writes the loss to tensorboard
    writer.add_scalar('Loss/train', np.mean(losses), epoch)

    return np.mean(losses), np.mean(losses_normal), np.mean(losses_prl), np.mean(losses_acl), opt, net
