"""
    This file test a network
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
import util_pytorch as up
from write_movie import write_movie, write_figure
'''
write_movie and write_figure taken from
https://github.com/hassanerrami/K_Enes/blob/0447b2115ae16edf8e07cae28bbc7425ffd8639c/RNN_baseline/write_movie.py
(unpublished)
written by Moritz Wolter 
'''
    
def test(net, data, mean, std, criterion, writer, epoch=0, debug=False, batch_size=50, device='cpu', pred_samples=50, write_vids=False, run=None, compression=None):
    '''
    Arguments
    ---------
    net: MoCapRNN network
    data: Human3.6M validation data to test the network
    mean: mean
    std: standard deviation
    criterion: loss criterion
    writer: tensorboard writer
    debug: debug mode on or off
    epoch: current epoch, only needed if debug == True
    batch_size: number of mini-sequences per mini-batch
    device: cuda or cpu
    pred_samples: number of samples to be predicted
    write_vids: True = write videos of the result
    run: number of run, only needed if write_vids == True
    compression: Yes or No, only needed if write_vids == True
    '''
    
    '''
    based on def train() from
    https://github.com/udacity/deep-learning-v2-pytorch/blob/3a95d118f9df5a86826e1791c5c100817f0fd924/recurrent-neural-networks/char-rnn/Character_Level_RNN_Solution.ipynb
    '''
    
    # absolute error
    abs_error = torch.nn.L1Loss()
    
    # initializes hidden state
    h = net.init_hidden(batch_size, device)

    # validation loss
    losses = []
    losses_normal = []
    losses_prl = []
    losses_acl = []
    losses_abs = []

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
        x = (x - mean) / std
        y = (y - mean) / std

        # ensures that input and target have the same data type
        inputs, targets = torch.from_numpy(x.astype('float32')), torch.from_numpy(y.astype('float32'))
        
        # puts all parameters on the GPU if possible
        inputs, targets, h = inputs.to(device), targets.to(device), h.to(device)

        # the following code line is needed, otherwise would backprop through
        # the entire training history of all batches of this epoch
        h = h.data

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
            loss = loss_normal
            losses_prl.append(0.)
            losses_acl.append(0.)
        else:
            loss = loss_normal + prl + acl
            losses_prl.append(prl.item())
            losses_acl.append(acl.item())
        loss_abs = abs_error(output, targets)
        
        # collects the loss
        losses_normal.append(loss_normal.item() * std + mean)
        losses.append(loss.item() * std + mean)
        losses_abs.append(loss_abs.item() * std + mean)
    
    if write_vids == False:    
        # writes the validation loss to tensorboard                 
        writer.add_scalar('Loss/test', np.mean(losses), epoch)
    
    # only of interest if vids of the test data are wished    
    elif write_vids == True:
        
        assert run != None
        assert compression != None
        
        # gets all stuff back on the cpu
        inputs, targets = inputs.to('cpu').numpy(), targets.to('cpu').numpy()
        output = output.to('cpu').detach().numpy()
        
        # reshapes them to get the coords in 3d back
        inputs = np.reshape(inputs, [inputs.shape[0], inputs.shape[1], 17, 3])
        targets = np.reshape(targets, [targets.shape[0], targets.shape[1], 17, 3])
        output = np.reshape(output, [output.shape[0], output.shape[1], 17, 3])
        
        # calculates the color shift for the vids
        # (point where the prediction starts)
        color_shift_at = len(inputs[0]) - 1
        
        # actual values
        original = np.concatenate((inputs, targets), 1)
        
        # predicted values
        predicted = np.concatenate((inputs, output), 1)
        
        # writes vids and pdfs
        for i in range(len(original)+1):
            write_movie(np.transpose(original[i], [1, 2, 0]), r_base=1000/std,
                    name='test_in_' + compression + '_' + str(run) + '_' + str(i) + '.mp4', color_shift_at=color_shift_at)
            write_movie(np.transpose(predicted[i], [1, 2, 0]), r_base=1000/std,
                        name='test_out_' + compression + '_' + str(run) + '_' + str(i) + '.mp4', color_shift_at=color_shift_at)        
            write_figure(np.transpose(original[i], [1, 2, 0]), r_base=1000/std,
                     name='test_in_' + compression + '_' + str(run) + '_' + str(i) + '.pdf', color_shift_at=color_shift_at)
            write_figure(np.transpose(predicted[i], [1, 2, 0]), r_base=1000/std,
                     name='test_out_' + compression + '_' + str(run) + '_' + str(i) + '.pdf', color_shift_at=color_shift_at)

    return np.mean(losses), np.mean(losses_normal), np.mean(losses_prl), np.mean(losses_acl), np.mean(losses_abs)
