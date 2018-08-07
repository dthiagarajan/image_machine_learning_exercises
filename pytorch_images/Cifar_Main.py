from pt_methods import *

batch_size = 128
initial_learning_rate = float(sys.argv[1])
if_aug = int(sys.argv[2])
'''
bank_L1_reg_rate = float(sys.argv[3])
bank_size = int(sys.argv[4])
key_size = int(sys.argv[5])
'''

data_path = "../Datasets"

# the normalized values are the usual values used by other implementations
if if_aug == 1:
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
                                          (0.2470, 0.2435, 0.2616))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616))])
else:
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))])
    transform_train = transform
    transform_test = transform

trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

net = WideResNet28_10()
# load model to resume training
#net = torch.load('pytorch_model.pt')
net.cuda()

'''
y_size = 10
net = DiffController(model=PreActResNet18, bank_size=bank_size,
                     key_size=key_size, y_size=y_size)
net.cuda()
'''

print(net)
#training(net, 200, initial_learning_rate, trainloader, testloader, test_device='gpu', fybl1lr=bank_L1_reg_rate, forget_option=False)
training(net, 200, initial_learning_rate, trainloader, testloader, test_device='gpu')
