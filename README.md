# image_machine_learning_exercises

My implementation of several popular deep learning image classification algorithms and applying them to Cifar-10 and medical images.

Some implementation ideas are learned from:
- https://github.com/jhkim89/PyramidNet
- https://github.com/hysts/pytorch_image_classification
- https://github.com/kuangliu/pytorch-cifar
- https://github.com/xternalz/WideResNet-pytorch

many thanks to them for sharing their code

The accuracies of PreActResNet, WideResNet and PyramidNet on the [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset matches the numbers in the paper.

On the [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) dataset (we use the 400X dataset), the accuracy is ~93-95% on testing data (traing/testing ratio 70%/30%) during multiple runs using PreActResNet18 as the model.

To run a model on the BreakHis dataset:
- 1 download the BreakHis dataset and put it in location ../Datasets
- 2 bash run_Net.sh
- 3 after the first run, you can change the run_Net.sh file to: python Histology_Main.py 1 0.5 0 0.1 1 1, which will directly load data from the npy files and save data reading time

To run the Cifar-10 dataset:
- 1 download the Cifar-10 dataset and put it in location ../Datasets
- 2 python Cifar_Main.py "initial learning rate" 1 (initial learning rate of 0.1 can replicate the accuracy mentioned above)
