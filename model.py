# -*- coding: utf-8 -*-
"""
Created on Mon May 17 22:40:38 2021

@author: user
"""

import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, hidden_cell_num, window_size, number_of_character):
        super(CharRNN, self).__init__()
        self.hidden_cell_num = hidden_cell_num
        self.rnn1 = nn.RNNCell(window_size, self.hidden_cell_num)
        self.rnn2 = nn.RNNCell(self.hidden_cell_num, self.hidden_cell_num)
        self.linear = nn.Lineaer(self.hidden_cell_num, number_of_character)
        
        
    def forward(self, dataset):

        batch_size = 128
        h_t, c_t, h_t2, c_t2 = self.init_hidden(batch_size)
        
        for data in dataset :
            h_t, c_t = self.rnn1(data, (h_t, c_t))
            h_t2, c_t2 = self.rnn2(h_t, h_t2, c_t2)
            output = self.linear(h_t2)
            
        return output

    def init_hidden(self, batch_size):

        h_t = torch.zeros(input.size(batch_size, self.hidden_cell_num, dtyepe=torch.float))
        c_t = torch.zeros(input.size(batch_size, self.hidden_cell_num, dtyepe=torch.float))
        h_t2 = torch.zeros(input.size(batch_size, self.hidden_cell_num, dtyepe=torch.float))
        c_t2 = torch.zeros(input.size(batch_size, self.hidden_cell_num, dtyepe=torch.float))
		
        return h_t, c_t, h_t2, c_t2

"""
class CharLSTM(nn.Module):
    def __init__(self):

        # write your codes here

    def forward(self, input, hidden):

        # write your codes here

        return output, hidden

		def init_hidden(self, batch_size):

				# write your codes here

				return initial_hidden
"""