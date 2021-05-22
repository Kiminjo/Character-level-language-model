# -*- coding: utf-8 -*-
"""
Created on Mon May 17 22:40:38 2021

@author: Kiminjo
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size=30, num_layer=1):
        super(CharRNN, self).__init__()
        
        # Use torh.nn.RNN
        # there is torch.nn.RNNCell but do not use this method
        # Check the below document if you want to know difference between RNN and RNNCell in pytorch
        # https://forum.onefourthlabs.com/t/difference-between-rnn-and-rnncell-in-pytorch/7643
        # RNN support multiple stacked layer but RNNCell does not
        # RNN needs several arguments
        # 1. input size : in this case, size of unique character number 59
        # 2. hidden size : hyperparameter, I set this 128
        # 3. bias
        # 4. num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.batch=100
        
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, bias=True, num_layers=self.num_layer, dropout=0.5, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.input_size)

           
    def forward(self, input, hidden):

        x, hidden = self.rnn(input, hidden)
        x = x.reshape(x.size(0) * x.size(1), x.size(2))
        output = self.linear(x)
            
        return output, hidden
    
    
    def init_hidden(self, batch_size):
        # there are 4 weight parameters in RNN 
        # input weight parameter W_x : size = (hidden_size, input_size)
        # hidden weight parameter W_h : size = (hidden_size, hidden_size)
        # bias input-hidden : size = (hidden)
        # bias hidden-hidden : size=(hidden)
        
        hidden_state = Variable(torch.zeros(self.num_layer, batch_size, self.hidden_size))
        
        return hidden_state




class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer):
        super(CharLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.batch = 100
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, bias=True, num_layers=self.num_layer, dropout=0.5, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.input_size)


    def forward(self, input, hidden):

        x, (hidden_state, cell_state) = self.lstm(input, hidden)
        x = x.reshape(x.size(0)*x.size(1), x.size(2))
        output = self.linear(x)
            
        return output, (hidden_state, cell_state)


    def init_hidden(self, batch_size):
        # there are 4 weight parameters in RNN 
        # input weight parameter W_x : size = (hidden_size, input_size)
        # hidden weight parameter W_h : size = (hidden_size, hidden_size)
        # bias input-hidden : size = (hidden)
        # bias hidden-hidden : size=(hidden)
        
        # LSTM needs hidden state and cell state 
        
        weight = next(self.parameters()).data
        hidden_state = weight.new(self.num_layer, batch_size, self.hidden_size).zero_()
        cell_state = weight.new(self.num_layer, batch_size, self.hidden_size).zero_()
        
        return (hidden_state, cell_state)
