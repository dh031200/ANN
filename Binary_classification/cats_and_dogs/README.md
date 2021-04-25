big data set

![basic_model](https://user-images.githubusercontent.com/66017052/115997403-f33c2680-a61d-11eb-8276-858d4712e500.png)

basic model (steps_per_epoch = 100, validation_steps=50)
  + train_acc: 0.8334375140815973
  + test_acc: 0.8400000159740448
  + elapsed time (in sec):  1132.4061875343323

tune batch_size
basic model (steps_per_epoch = 8000//32, validation_steps=2000//32)
  + train_acc: 0.9023750121891498
  + test_acc: 0.9000000128746033
  + elapsed time (in sec):  2273.6289598941803
  + score : 17.37183  (kaggle)

add 2 dropout layers
  + train_acc: 0.9183750104159116
  + test_acc: 0.9042000093460083
  + elapsed time (in sec):  2313.947970867157
  + score : 17.18115  (kaggle)

transfer learning
basic InceptionV3 model
  + train_acc: 0.9465625031292438
  + test_acc: 0.9542000005245209
  + elapsed time (in sec):  1476.256034374237

fine-tuned InceptionV3 model
  + train_acc: 0.9466875042021274
  + test_acc: 0.956400003194809
  + elapsed time (in sec):  16015.294801712036
