# ABC_Notation_Music_Generation


### Introduction
In this deep learning project, which is inspired by the Andrej Karpathy's blog 'The Unreasonable Effectiveness of Recurrent Neural Networks',  I implemented a LSTM model on abc notation music data.

### Requirements
This repository requires installation of Tensorflow, Numpy and Keras to run smoothly.

### Information regarding this repository
The input data is present in './data' named 'input.txt'.

The accuracy and loss associated with every epoch is stored in './logs' in csv format named 'training_log.csv'.

Model at every epoch is stored in './model' in h5 format.

### How to use
Use 'sample.py' to run the trained model with no number of epochs, length and seed as arguments to the program.

Use 'train.py' to train the model from scratch. Here the arguments are input, epochs and save_freq.

Use 'model.py' to change the parameters of the model.
