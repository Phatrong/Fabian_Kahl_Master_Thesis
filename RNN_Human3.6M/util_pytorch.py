"""
    This file contains some utils
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

def organize_into_batches(data, batch_size):
    '''
    Arguments
    ---------
    data: input data
    batch_size: wished batch size
    '''
    
    '''
    based on def organize_into_batches() from
    https://github.com/hassanerrami/K_Enes/blob/0447b2115ae16edf8e07cae28bbc7425ffd8639c/RNN_baseline/util.py
    (unpublished)
    written by Moritz Wolter
    '''
    
    batch_total = len(data)
    split_into = int(batch_total/batch_size)
    
    # stop position for the last possible full batch within the data
    stop_at = batch_size*split_into
    
    # spilts the data into batches of length batch_size each
    batch_lst = np.array_split(np.stack(data[:stop_at]), split_into)
    return batch_lst


def get_input_target_batches(data, pred_samples):
    '''
    Arguments
    ---------
    data: data to be split in inputs (x) and targets (y) for the training step
    pred_samples: number of samples to be predicted
    '''
    
    h_help = np.copy(data)
    h_help = np.transpose(h_help, (2,1,0,3,4))
    h_length = len(h_help)
    
    # cut for the input data
    cut = h_length - pred_samples
    
    x = h_help[:cut]
    y = h_help[pred_samples:]
    
    x = np.transpose(x, (2,1,0,3,4))
    y = np.transpose(y, (2,1,0,3,4))
    return x, y