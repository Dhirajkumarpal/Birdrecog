#**Bird Image Recognition**

Team Members
    1. Prithvi Kunder
    2. Rohit Nadiger
    3. Dhirajkumar Pal
    
We have created a bird image classifier(which will classify whether the given image consists of a bird or not).Our classifier is a  Convolutional Neural Networks(CNN). We have prepared this classifier using Tensorflow and Tflearn deep learning packages.

We have trained our Neural Network on Google Colaboratory. Project contains two python files:

    1. train_bird.py : Model consists of three hidden layers which uses relu activation function and output layer which uses                softmax activation function.
    Checkpoints are saved in the file(bird-classifier.tfl).
    
    2. test_bird.py: Checkpoints are loaded and model performs predictions on input image present in test_images directory.
