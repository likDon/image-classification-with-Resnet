# image-classification-with-Resnet
You can modify Data Augmentation method, Net Architecture and Training method.
- **Data Augmentation**: We offer 3 methods: Simple Augmentation(including flip and clip), Cutmix and Cutout 
- **Net Architecture**: Set C1, layer number and block number of each layer.
- **Training method**: Set batch size, epoch number, optimizer, learning rate and lr_scheduler

```
BATCH_SIZE = 128
EPOCH = 200
LR = 0.1
C1 = 31
Net_Num_Blocks = [3,4,6,3]

Data_Aug = 'cutmix' # 'cutmix', 'cutout','simple'
Optim_type = 'momentum' #'SGD', 'momentum', 'Adam'
Scheduler_type = 'MultiStep' #'None', 'CLR', 'MultiStep'
```
Using Resnet34 and CutMix(settings are shown above), our final accuracy on CIFAR10 is **91.8%**.(Model size is 4.9M)
