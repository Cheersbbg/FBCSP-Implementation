# Filterbank-CSP Implementation in Python


<span style="font-size:10em;"> Filterbank CSP is a common benchmark used during comparison of novel methods of motor imagery classification research. However, many currently existing implementations of it are rather slow, do not work properly, or do not use the original data sources of the BCI Competition IV Dataset II a and II b motor imagery training and evaluation trials, making replicability very difficult, and a lot of work in papers just going towards re-establishing a baseline model. This repository is meant to help with this. </span>



### Usage

First, make sure to download the training and evaluation files from BCI Competition IV IIa with this link: https://bbci.de/competition/iv/download/

And then download the labels of the evaluation datasets here: https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip

By default when loading motor imagery trials from the GDF Files, the default parameter for the directory in which they are stored in is called "GDF Files". This may be changed with the "data_dir" keyword parameter. 


When loading in data with the "load_motor_imagery_trials" make sure to specify which directory the GDF files are in with the "data_dir" key word command. 

With the "load_evaluation_trials" function please also specifiy which directory the evaluation trials, and the evaluation labels are in with the data_dir and labels_dir keyword parameters. 



The requirements.txt file is included so that you can use the same conda environment I was using while this code was written. 


To see how the code is all used together, please see "Model Validation.ipynb"


