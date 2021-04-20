MNIST ANN
+ 1 hidden layer with 512 neurons, use relu activation
+ test_acc:  0.9807999730110168
+ elapsed time (in sec):  16.526145696640015 (early stopping at epoch 9/20)

MNIST CNN
+ 2 Convolution layers with 32, 64 neurons, use relu activation
+ 1 Poolinglayer (Maxpooling)
+ MLP
+ test_acc:  0.9894000291824341 
+ elapsed time (in sec):  145.70548224449158 (early stopping at epoch 4/20)

Easy datasets such as MNIST show high accuracy even with simple models.
It seems better to use ANN than CNN, which are time-consuming due to their complexity.
