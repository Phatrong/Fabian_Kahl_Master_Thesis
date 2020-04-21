"""
    This file contains RNN structures for the Human3.6M dataset
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

from torch import nn
import torch
import pywt
from wavelet_learning.wavelet_linear import WaveletLayer
'''
WaveletLayer taken from
https://github.com/v0lta/waveletnet/blob/dc7252bf7991b19fdf8f238dcd228f4eb039d97f/wavelet_learning/wavelet_linear.py
(unpublished)
written by Moritz Wolter
'''

class MoCapRNN(nn.Module):
'''
based on class GRUCell() from
https://github.com/v0lta/waveletnet/blob/bcf77ba84239bc1dc58e7edcef55271fe0cd889b/RNN_compression/cells.py
(unpublished)
written by Mority Wolter
'''
  
    def __init__(self, n_input=17*3, n_hidden=1024*3):
        '''
        Arguments
        ---------
        n_input: number of input units
        n_hidden: number of hidden units per layer
        '''

        super().__init__()
        self.n_hidden = n_hidden
        self.n_input = n_input
        
        # GRU
        # =====================================================================
        # update gate weights
        self.Whz = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Wxz = nn.Linear(n_input, n_hidden, bias=True)

        # reset gate weights
        self.Whr = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Wxr = nn.Linear(n_input, n_hidden, bias=True)

        # hidden state weights
        self.Whh = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Wxh = nn.Linear(n_input, n_hidden, bias=True)
        # =====================================================================

        # fully-connected output layer
        self.W_proj = nn.Linear(n_hidden, n_input, bias=True)
        
    
    def forward(self, x, h):        
        '''
        Arguments
        ---------
        x: network input
        h: hidden state
        '''
        
        # removes single-dimensional entries from the shape of h
        h = torch.squeeze(h)
        # applies GRU formulas
        z = torch.sigmoid(self.Whz(h) + self.Wxz(x))
        r = torch.sigmoid(self.Whr(h) + self.Wxr(x))
        hc = torch.tanh(self.Whh(r*h) + self.Wxh(x))
        hn = (1 - z)*h + z*hc
        y = self.W_proj(hn)
        
        # residual connection
        y += x

        # returns the final output and the hidden state
        return y, hn
    
    
    def get_wavelet_loss(self):
        # in a non-compressed network, no wavelet_loss exists
        return -1,-1

    
    def init_hidden(self, batch_size, device):
        '''
        Arguments
        ---------
        batch_size: number of mini-sequences per mini-batch
        device: cuda or cpu
        '''
        
        # Creates one new tensor with size 1 x batch_size x n_hidden,
        # initialized to zero, for hidden state of GRU
        hidden  = torch.zeros(1, batch_size, self.n_hidden)
        if device == 'cuda' :
             hidden  = hidden.cuda()
        return hidden
    

class MoCapRNNWavelet(MoCapRNN):
'''
based on class WaveletGRU() from
https://github.com/v0lta/waveletnet/blob/bcf77ba84239bc1dc58e7edcef55271fe0cd889b/RNN_compression/cells.py
(unpublished)
written by Moritz Wolter
'''

    def __init__(self, n_input=17*3, n_hidden=1024*3, 
                 init_wavelet=pywt.Wavelet('db6'), mode='full'):
        super().__init__(n_input, n_hidden)
        self.init_wavelet = init_wavelet
        self.mode = mode
        scales = 6
        if mode == 'gates':
            print('gates compression')
            self.Whz = WaveletLayer(n_hidden, init_wavelet=init_wavelet, scales=scales, p_drop=0.0)
            self.Whr = WaveletLayer(n_hidden, init_wavelet=init_wavelet, scales=scales, p_drop=0.0)
        elif mode == 'reset':
            print('reset compression')
            self.Whr = WaveletLayer(n_hidden, init_wavelet=init_wavelet, scales=scales, p_drop=0.0)
        elif mode == 'update':
            print('update compression')
            self.Whz = WaveletLayer(n_hidden, init_wavelet=init_wavelet, scales=scales, p_drop=0.0)
        elif mode == 'state':
            print('state compression')
            self.Whh = WaveletLayer(n_hidden, init_wavelet=init_wavelet, scales=scales, p_drop=0.0)
        elif mode == 'state_reset':
            print('state+reset gate compression')
            self.Whh = WaveletLayer(n_hidden, init_wavelet=init_wavelet, scales=scales, p_drop=0.0)
            self.Whr = WaveletLayer(n_hidden, init_wavelet=init_wavelet, scales=scales, p_drop=0.0)
        elif mode == 'state_update':
            print('state+update gate compression')
            self.Whh = WaveletLayer(n_hidden, init_wavelet=init_wavelet, scales=scales, p_drop=0.0)
            self.Whz = WaveletLayer(n_hidden, init_wavelet=init_wavelet, scales=scales, p_drop=0.0)
        else:
            print('full compression')
            self.Whz = WaveletLayer(n_hidden, init_wavelet=init_wavelet, scales=scales, p_drop=0.0)
            self.Whr = WaveletLayer(n_hidden, init_wavelet=init_wavelet, scales=scales, p_drop=0.0)
            self.Whh = WaveletLayer(n_hidden, init_wavelet=init_wavelet, scales=scales, p_drop=0.0)
        print('Creating a Wavelet GRU, do not forget to add the wavelet-loss.')

    def get_wavelet_loss(self):
        # prl is the perfect reconstruction loss
	# acl is the alias cancellation loss aka anti-aliasing loss
        if self.mode == 'gates':
            Whz_prl, Whz_acl = self.Whz.get_wavelet_loss()
            Whr_prl, Whr_acl = self.Whr.get_wavelet_loss()
            return Whz_prl + Whr_prl, Whz_acl + Whr_acl
        elif self.mode == 'state':
            Whh_prl, Whh_acl = self.Whh.get_wavelet_loss()
            return Whh_prl, Whh_acl
        elif self.mode == 'reset':
            Whr_prl, Whr_acl = self.Whr.get_wavelet_loss()
            return Whr_prl, Whr_acl
        elif self.mode == 'update':
            Whz_prl, Whz_acl = self.Whz.get_wavelet_loss()
            return Whz_prl, Whz_acl
        elif self.mode == 'state_reset':
            Whh_prl, Whh_acl = self.Whh.get_wavelet_loss()
            Whr_prl, Whr_acl = self.Whr.get_wavelet_loss()
            return Whh_prl + Whr_prl, Whh_acl + Whr_acl
        elif self.mode == 'state_update':
            Whh_prl, Whh_acl = self.Whh.get_wavelet_loss()
            Whz_prl, Whz_acl = self.Whz.get_wavelet_loss()
            return Whh_prl + Whz_prl, Whh_acl + Whz_acl
        else:
            Whh_prl, Whh_acl = self.Whh.get_wavelet_loss()
            Whz_prl, Whz_acl = self.Whz.get_wavelet_loss()
            Whr_prl, Whr_acl = self.Whr.get_wavelet_loss()
            return Whh_prl + Whz_prl + Whr_prl, Whh_acl + Whz_acl + Whz_acl
