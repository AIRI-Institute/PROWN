from typing import Any

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import nn
from torchvision.models import densenet201

from models import resnet34

from utils import steal_train, get_dataset, evaluate_model

from absl import app, flags

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    ])

transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    ])

FLAGS = flags.FLAGS
def main(argv):
    del argv

    batch_size = FLAGS.batch_size
    num_epochs = FLAGS.num_epochs    
    num_models = FLAGS.num_models
    save_path = FLAGS.save_path
    dataset = FLAGS.dataset

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Available Device ', device)

    trainset, testset = get_dataset(dataset)
    model_path = FLAGS.target_path
    model = resnet34(pretrained=False)

    data = torch.load(model_path)['model']
    model.to(device)
    model.load_state_dict(data)


    trainset_cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset_cifar10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    

    trainloader_cifar10 = DataLoader(trainset_cifar10, batch_size=512, shuffle=True)
    testloader_cifar10 = DataLoader(testset_cifar10, batch_size=512, shuffle=True)

    print(f'Teacher model evaluation')
    teacher_train_acc = evaluate_model(model, trainloader_cifar10, device)
    teacher_test_acc = evaluate_model(model, testloader_cifar10, device)
    print(f'Train Acc {teacher_train_acc:.3f} Test Acc {teacher_test_acc:.3f}')

    for i in range(num_models):
        temp_model = densenet201(pretrained=False)
        temp_model.classifier = nn.Linear(in_features=1920, out_features=10)
        # temp_model.classifier[6] = nn.Linear(in_features=4096, out_features=10)
        steal_train(model, temp_model, trainset, testset, num_epochs, device, 
                    do_eval=True, epoch_eval=5, batch_size=batch_size, savedir=save_path + 'model_student_' + str(i+1), save_iter=10, hard_label=False)
        train_acc_cifar10 = evaluate_model(temp_model, trainloader_cifar10, device)
        test_acc_cifar10 = evaluate_model(temp_model, testloader_cifar10, device)
        print(f'Model {i+1} train. Train acc {train_acc_cifar10:.3f} Test acc {test_acc_cifar10:.3f}')

if __name__ == '__main__':
    flags.DEFINE_integer('batch_size', 512, 'Batch size')
    flags.DEFINE_integer('num_epochs', 1000, 'Training duration in number of epochs.')
    flags.DEFINE_integer('num_models', 10, 'Amount of models to be trained')
    flags.DEFINE_boolean('random_transform', False, 'Using random transform')
    flags.DEFINE_string('target_path', 'model_0.pt', 'Path to target model')
    flags.DEFINE_string('save_path', './models/densenet201_torchvision/', 'Path to save model')
    flags.DEFINE_string('dataset', 'cifar100', 'Dataset for model stealing')    
    
    app.run(main)
