# -*- coding: utf-8 -*-
"""
Created on Mon May 17 22:40:56 2021

@author: Kiminjo
"""

import torch
from torch.utils.data import DataLoader
from dataset import Shakespeare, one_hot_encoding
from model import CharRNN, CharLSTM
from generate import generate
import warnings 
warnings.filterwarnings(action='ignore')

def train(model, trn_loader, device, criterion, optimizer, batch_size, network_type):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    model.train()
    
    total_batch = len(trn_loader)
    trn_loss = 0
    
    hidden = model.init_hidden(batch_size)
    
    for batch_idx, batch in enumerate(trn_loader) :
        x, label = batch

        # input sequence x should be form of one hot vector 
        x = one_hot_encoding(x)
        x = x.to(device); label = label.to(device)
        if network_type=='RNN' :
            hidden = tuple([each.data for each in hidden])[0].reshape(1, -1, 512)
        else :
            hidden = tuple([each.data for each in hidden])
        optimizer.zero_grad()
        output, hidden = model.forward(x, hidden)
        cost = criterion(output, label.view(3000).long())
        cost.backward(retain_graph=True)
        optimizer.step()
        
        trn_loss += cost.item()
        
    trn_loss = round(trn_loss/total_batch, 3) 

    return trn_loss


@torch.no_grad()
def validate(model, val_loader, device, criterion, batch_size, network_type='RNN'):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    model.eval()
    
    total_batch = len(val_loader)
    val_loss = 0
    
    hidden = model.init_hidden(batch_size)
    
    for batch_idx, batch in enumerate(val_loader) :
        x, label = batch

        # input sequence x should be form of one hot vector 
        x = one_hot_encoding(x)
        x = x.to(device); label = label.to(device)
        if network_type=='RNN' :
            hidden = tuple([each.data for each in hidden])[0].reshape(1, -1, 512)
        else :
            hidden = tuple([each.data for each in hidden])
        output, hidden = model.forward(x, hidden)
        cost = criterion(output, label.view(3000).long())
        
        val_loss += cost.item()

    val_loss = round(val_loss/total_batch, 3) 

    return val_loss


def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    
    input_file = open('data/shakespeare_train.txt', 'r').read()
    epochs = 3
    batch_size = 100
    
    train_dataset = Shakespeare(input_file, is_train=True)
    test_dataset = Shakespeare(input_file, is_train=False)
    
    train_data = DataLoader(train_dataset, batch_size=batch_size)
    test_data = DataLoader(test_dataset, batch_size=batch_size)
    
    ##################################################################
    #                   LSTM model
    ##################################################################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lstm_model = CharLSTM(input_size=len(train_dataset.char2int), 
                    hidden_size=512, num_layer=1).to(device)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)
    lstm_criterion = torch.nn.CrossEntropyLoss().to(device)
    
    lstm_trn_loss = []; lstm_val_loss = []
    for epoch in range(epochs) :
        train_loss = train(lstm_model, train_data, device, lstm_criterion, lstm_optimizer, batch_size, 'LSTM')
        test_loss = validate(lstm_model, test_data, device, lstm_criterion, batch_size, 'LSTM')
        print('epoch : {}, train loss : {}, validation loss : {}'.format(epoch+1, train_loss, test_loss))
        
        lstm_trn_loss.append(train_loss); lstm_val_loss.append(test_loss)
        
    
    
    ##################################################################
    #                   RNN model
    ##################################################################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rnn_model = CharRNN(input_size=len(train_dataset.char2int), 
                    hidden_size=512, num_layer=1).to(device)
    rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.01)
    rnn_criterion = torch.nn.CrossEntropyLoss().to(device)
    
    rnn_trn_loss = []; rnn_val_loss = []
    for epoch in range(epochs) :
        train_loss = train(rnn_model, train_data, device, rnn_criterion, rnn_optimizer, batch_size, 'RNN')
        test_loss = validate(rnn_model, test_data, device, rnn_criterion, batch_size, 'RNN')
        print('epoch : {}, train loss : {}, validation loss : {}'.format(epoch+1, train_loss, test_loss))
        
        rnn_trn_loss.append(train_loss); rnn_val_loss.append(test_loss)
        
    
    lstm_generated_text = generate(lstm_model, 'The', 5, 'LSTM', train_dataset.char2int, train_dataset.int2char)
    #rnn_generated_text = 
        

    return lstm_trn_loss, lstm_val_loss, lstm_generated_text



if __name__ == '__main__':
    train_loss, val_loss, generated_text = main()