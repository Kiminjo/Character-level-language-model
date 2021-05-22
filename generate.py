# -*- coding: utf-8 -*-
"""
Created on Mon May 17 22:41:22 2021

@author: Kiminjo
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataset import one_hot_encoding

def generate(model, seed_characters, temperature, network_type, char2int_dict, int2char_dict):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
				temperature: T
				args: other arguments if needed

    Returns:
        samples: generated characters
    """

    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    samples = [string for string in seed_characters]
    hidden = model.init_hidden(1)
    
    for string in seed_characters :
        x = np.array([[char2int_dict(string)]])
        x = one_hot_encoding(x)
        x = torch.from_numpy(x).to(device)
        
        if network_type == 'RNN' :
            hidden = tuple([each.data for each in hidden])[0].reshape(1, -1, 512)
        else :
            hidden = tuple([each.data for each in hidden])
            
        output, hidden = model.forward(x, hidden)
        
        prob = F.softmax(output, dim=1).data
        prob, top_char = prob.topk(temperature)
        top_char = top_char.numpy().squeeze()
        prob = prob.numpy().squeeze()
        
        character = np.random.choice(top_char, p=prob/prob.sum())
        character = int2char_dict[character]
        
    samples.append(character)
        
    return samples