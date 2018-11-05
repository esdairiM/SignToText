# SignToText

This is my approach to American Sign Language Recognition (Alphabet level only) using MNIST ASL Dataset and a small ConvNet, The goal here was to train a model with high accuracy using CPU only on a small dataset.
For simplicity, I used Keras with a TensorFlow backend, I also used Numpy for matrix manipulation, Pandas to read the data properly and Matplotlib for plots and visualization.

## Data

the two CSV files contain all the image data, each row is basically a flattened image, there is a train CSV for training the model (train + validation) and a test CSV to evaluate how the model does on unseed data.

##  To run the model:

* Download the 2 CSV files, and the .ipynb file. 
* make sure you have the latest TensroFlow version installed, this notebook was created using this docker image :  
tensorflow/tensorflow:nightly-devel-py3
* make sure you have Numpy, Pandas, and Matplotlib

## If you just want to see the results

* access "CNNonASLofMNIST.md" this file will show you a nice version of the notebook.
