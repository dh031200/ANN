![Accuracy](https://user-images.githubusercontent.com/66017052/115433993-2314b400-a243-11eb-8b51-b7b36e87f97a.png)

Pretrained model
+ use VGG19
+ epoch 50
+ early stopping patience=3
+ train_acc:  0.9885714054107666
+ test_acc:  0.6875
+ elapsed time (in sec):  141.11554384231567

Transfer learning model
+ load pretrained model
+ trainable layers with start block5 
+ epoch 50
+ early stopping patience=3
+ train_acc:  0.9971428513526917
+ test_acc:  0.7321428656578064
+ elapsed time (in sec):  313.96853137016296
