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
        self.encoded, self.char_dict = integer_encoding(input_file)
        self.input, self.target = generate_mini_batch(self.encoded, self.sequence_length)

               
        if is_train == True : 
            self.input = self.input[: int(self.input.size(0) * 0.9)]
            self.target = self.target[: int(self.target.size(0) * 0.9)]
        else :
            self.input = self.input[int(self.input.size(0) * 0.9) : ]
            self.target = self.target[int(self.target.size(0) * 0.9) : ]
        

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
    character_dict = {char : idx for idx, char in enumerate(characters)}
    encoded_text = list(map(character_dict.get, text_file, text_file))
    character_dict = dict((idx, char) for char, idx in character_dict.items())
    
    return encoded_text, character_dict
    


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
    

if __name__ == '__main__':

    input_file = open('data/shakespeare_train.txt', 'r').read()
    train_dataset = Shakespeare(input_file, is_train=True)
    test_dataset = Shakespeare(input_file, is_train=False)
    
    train = DataLoader(train_dataset, batch_size=128)
    