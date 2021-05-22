# -*- coding: utf-8 -*-
"""
Created on Mon May 17 22:39:52 2021

@author: Kiminjo
"""

import torch
from torch.utils.data import Dataset, DataLoader

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
					 You need this dictionary to generate characters.
				2) Make list of character indices using the dictionary
				3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file, is_train=True):
        # get integer mapped sequence data and integer-character dictionary using 'integer_encoding' function
        # use capital and punctuation in this work. 
        # More detail, read this document 
        # https://www.researchgate.net/publication/319185593_Experiments_in_Character-Level_Neural_Network_Models_for_Punctuation
        self.sequence_length = 30
        self.encoded, self.int2char, self.char2int = integer_encoding(input_file)
        self.input, self.target = generate_mini_batch(self.encoded, self.sequence_length)

               
        if is_train == True : 
            self.input = remove_some_data(self.input[: int(self.input.size(0) * 0.9)])
            self.target = remove_some_data(self.target[: int(self.target.size(0) * 0.9)])
        else :
            self.input = remove_some_data(self.input[int(self.input.size(0) * 0.9) : ], is_train=False)
            self.target = remove_some_data(self.target[int(self.target.size(0) * 0.9) : ], is_train=False)
        

    def __len__(self):
        return len(self.input)
    

    def __getitem__(self, idx):
        
        input = self.input[idx]
        target = self.target[idx]

        return input, target
    
    
 
def integer_encoding(input_text) :
    text_file = []
    
    # remove chracter name in text
    # for example, the text "injo : hello, world" leaves only "hello, world" 
    input_text = ''.join([word for word in list(filter(None, input_text.splitlines())) if ':' not in word])
    text_file += [words for words in input_text] 
    
    # get unique character to make integer-character dictionary
    characters = list(set(text_file))
    
    # normally, dictionary is {index : character} it has to be made in order
    # But in this task, we should maps character to indecies so it changes the key and value of the dictionary after making them into {character : index}
    char2int = {char : idx for idx, char in enumerate(characters)}
    encoded_text = list(map(char2int.get, text_file, text_file))
    encoded_text = encoded_text
    int2char = dict((idx, char) for char, idx in char2int.items())
    
    return encoded_text, int2char, char2int
    


def generate_mini_batch(data, sequence_length) :
    # Cut sequence data into 30 chunks
    # The input data and target data are each composed of 30 sequence
    # The target at point t is the same as the input at point t+1     
    input = []; target = []
    for index in range(len(data) - 1) :
        if index >= sequence_length -1  :
            row_x = [] ; row_y = []
            row_x += [data[index-idx] for idx in range(sequence_length)]
            row_x.reverse(); row_y = row_x.copy()
            row_y.pop(0); row_y.append(data[index+1])
            input.append(row_x); target.append(row_y)
    
            
    return torch.tensor(input, dtype=torch.float), torch.tensor(target, dtype=torch.float)
    

def one_hot_encoding(input_data) :
    # input dimension is 2-dimensional matrix
    # but RNN got 3 dimensional tensor as input 
    # so change the 2 dimension matrix to three dimensional tensor using one hot encoding
    dict_size = 59
    output = []

    
    for row in input_data.split(1) :
        row = row[0]
        row_to_matrix = []
        for x in row :
            one_hot_vector = torch.eye(dict_size, dtype=torch.float)[int(x)]
            one_hot_vector = one_hot_vector.tolist()
            row_to_matrix.append(one_hot_vector)
        
        output.append(row_to_matrix)
    output = torch.tensor(output)
    return output


def remove_some_data(x, is_train=True) :
    if is_train==True :
        remove_data = 38
    else :
        remove_data = 27
    x = x[:-remove_data]
    return x


if __name__ == '__main__':

    input_file = open('data/shakespeare_train.txt', 'r').read()
    train_dataset = Shakespeare(input_file, is_train=True)
    test_dataset = Shakespeare(input_file, is_train=False)
    
    train_data = DataLoader(train_dataset, batch_size=100)
    test_data = DataLoader(test_dataset, batch_size=100)
    print(len(train_dataset))
    print(len(train_data))
    