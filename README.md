# Character Level Langauge Model

It aims to write new sentences by learning character units sentences using RNN. As training data, a collection of Shakespeare's novels was used. This project was carried out as part of the 'Artificial Neural Networks and Deep Learning' class in the Spring Semester of Data Science at Seoul National University of Science and Technology in 2021.

<br></br>

## Character level langauge model

It has the same structure as the existing RNN model, but has one alphabetic character, not a token, as an input. At this time, characters include spaces, paragraph changes, etc.

The structure of the character-level language model is shown in the figure below.

![structure](https://user-images.githubusercontent.com/42087965/140010696-65ff1b41-8a6c-4716-b1a2-4d28c1c580bb.png)

It takes a specific starting token as input and outputs a character that is likely to appear later. The output character is a recursive model that goes back into the input of the model and predicts the next character.

Please check [here](https://towardsdatascience.com/character-level-language-model-1439f5dd87fe), if you interested in character level language model.

<br></br>

## Dataset

The Shakespear's Literature Collection was used as the experimental dataset. It takes the form of a play of "Speaker: Dialogue", and all of them are used as input data. Thus, the newly created text also took the form of a play script.

In addition, paragraphs change for every line, and this was also learned with input data.

The training data used in the experiment can be found [here](https://github.com/Kiminjo/Character-level-language-model/blob/main/data/shakespeare_train.txt).

<br></br>

## Software Requirements

- python >= 3.5
- pytorch
- numpy 
- matplotlib

<br></br>

## Key Files

- `dataset.py` : It takes text data as input and converts it into a batch of tensor.
- `model.py` : A model used for training and sentence generation was defined. Two basic RNN models and LSTM models were used in the experiment.
- `main.py` : The main file of this project. Train the our character langauge model. In addition, the error rate was visualized using matplotlib.
- `generate.py` : A new sentence is generated using the model learned in the main file.
