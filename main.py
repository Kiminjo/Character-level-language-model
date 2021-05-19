# -*- coding: utf-8 -*-
"""
Created on Mon May 17 22:40:56 2021

@author: user
"""

import torch
from torch.utils.data import DataLoader
from dataset import Shakespeare
from model import CharRNN

# import some packages you need here


def train(model, trn_loader, device, criterion, optimizer):
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
    
    for batch_idx, batch in enumerate(trn_loader) :
        seq, label = batch
        seq = seq.to(device); label = label.to(device)
        optimizer.zero_grad()
        output = model(seq)
        cost = criterion(output, label)
        cost.backward()
        optimizer.step()
        
        trn_loss += cost.itme()
        
    trn_loss = round(trn_loss/total_batch, 3) 
    

    return trn_loss


def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    # write your codes here

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
    epochs = 10
    batch_size = 128
    
    train_dataset = Shakespeare(input_file, is_train=True)
    test_dataset = Shakespeare(input_file, is_train=False)
    
    train = DataLoader(train_dataset, batch_size=batch_size)
    test = DataLoader(test_dataset, batch_size=batch_size)
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CharRNN(hidden_cell_num=batch_size, window_size=30, number_of_character=len(train_dataset.character_dict)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    for epoch in range(epochs) :
        train_loss = train(model, train, device, criterion, optimizer)
        
    

    # write your codes here

if __name__ == '__main__':
    main()