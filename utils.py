import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
import torchvision.datasets as datasets

def train(model, train_set, flip_set, test_set, num_epochs, device, do_eval=False, n_flip_train_epoch = 5, epoch_eval=5, batch_size=10, criterion=None, opt=None):
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if opt is None:
            opt = optim.Adam(model.parameters())
            
        model = model.to(device)  
         
        train_loss_history = []
        test_loss_history = []
        flip_loss_history = []
        train_acc_history = []
        train_f1_history = []
        flip_acc_history = []
        test_acc_history = []
        flip_acc_history = []
        test_f1_history = []
        flip_f1_history = []
        
        train_dataloader = DataLoader(train_set, batch_size=256, shuffle=True)
        flip_dataloader = DataLoader(flip_set, batch_size=256, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=256, shuffle=True)
        
        for epoch in tqdm(range(num_epochs)):  # Loop over the dataset multiple times
            # Train on trainset
            running_loss = 0.0
            total_train = 0
            correct_train = 0
            f1_train = 0.0

            model.train()
            for i, data in enumerate(train_dataloader, 0):
                inputs_inter, labels = data
                inputs_inter = inputs_inter.float().to(device)
                labels = labels.to(device)

                opt.zero_grad()
                outputs = model(inputs_inter)
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()

                _, predicted = torch.max(outputs.data, 1)

                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                f1_train += f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')

                running_loss += loss.item()

            train_loss_history.append(running_loss / len(train_dataloader))
            train_acc_history.append(correct_train / total_train)
            train_f1_history.append(f1_train / len(train_dataloader))


            # Train on flip set
            running_loss = 0.0
            total_flip = 0
            correct_flip = 0
            f1_flip = 0.0

            model.train()
            for j in range(n_flip_train_epoch):
                for i, data in enumerate(flip_dataloader, 0):
                    inputs_inter, labels = data
                    inputs_inter = inputs_inter.float().to(device)
                    labels = labels.to(device)

                    opt.zero_grad()
                    outputs = model(inputs_inter)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    opt.step()

                    _, predicted = torch.max(outputs.data, 1)

                    total_flip += labels.size(0)
                    correct_flip += (predicted == labels).sum().item()
                    f1_flip += f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')

                    running_loss += loss.item()

                flip_loss_history.append(running_loss / len(flip_dataloader))
                flip_acc_history.append(correct_flip / total_flip)
                flip_f1_history.append(f1_flip / len(flip_dataloader))
                print(f'Train acc {train_acc_history[-1]}, Flip acc {flip_acc_history[-1]}')
            if do_eval:
                if epoch % epoch_eval == 0:  # Compute and plot test metrics every `test_plot_iters` epochs
                    model.eval()  # Set model to evaluation mode
                    total_test = 0
                    correct_test = 0
                    f1_test = 0.0
                    test_loss = 0.0

                    with torch.no_grad():  # Deactivate gradients for the following code block
                        for data in test_dataloader:
                            inputs_inter, labels = data
                            inputs_inter = inputs_inter.float().to(device)
                            labels = labels.to(device)
                            outputs = model(inputs_inter)
                            _, predicted = torch.max(outputs.data, 1)
                            loss = criterion(outputs, labels)
                            test_loss += loss.item()
                            total_test += labels.size(0)
                            correct_test += (predicted == labels).sum().item()
                            f1_test += f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')

                    test_acc_history.append(correct_test / total_test)
                    test_f1_history.append(f1_test / len(test_dataloader))
                    test_loss_history.append(test_loss / len(test_dataloader))
                    print(f'Test acc {test_acc_history[-1]}')
            else:
                pass
            

        return train_acc_history, flip_acc_history, test_acc_history
        

def flip_labels(dataset, part=0.1, indices=None):
    n_samples = len(dataset)
    n_flip = int(n_samples * part)

    if indices is None:
        indices = np.random.choice(n_samples, n_flip, replace=False)

    for idx in indices:
        dataset.targets[idx] = (dataset.targets[idx] + 1) % 10

    return indices


def evaluate_model(model, dataloader, device):
    model.eval()
    model.to(device)
    correct = 0.0
    total = 0.0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0], data[1]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def add_uniform_noise(model, eps):
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.empty_like(param).uniform_(-eps, eps)
            param.add_(noise)


def steal_train(target_model, model, trainset, testset, num_epochs, device, do_eval=False, epoch_eval=5, batch_size=10, opt=None, savedir='./model.pt', save_iter=10, hard_label=False):

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

    if hard_label:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.KLDivLoss(reduction='batchmean')
    if opt is None:
        opt = optim.Adam(model.parameters(), lr=10e-3)
            
    model = model.to(device)  
    target_model = target_model.to(device)

    train_loss_history = []
    test_loss_history = []

    train_student_acc = []
    test_student_acc = []

    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    # Evaluate Teacher model
    train_teacher_acc = evaluate_model(target_model, train_dataloader, device)
    test_teacher_acc = evaluate_model(target_model, test_dataloader, device)
    print(f'Evaluated Teacher Model')
    print(f'Teacher Model Train Acc {train_teacher_acc:.3f}, Test Acc {test_teacher_acc:.3f}')

    trainset_cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset_cifar10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    

    trainloader_cifar10 = DataLoader(trainset_cifar10, batch_size=512, shuffle=True)
    testloader_cifar10 = DataLoader(testset_cifar10, batch_size=512, shuffle=True)

    for epoch in tqdm(range(num_epochs)):

        if epoch % save_iter == 0:
            print('Model saved, CIFAR10 performance:')
            
            train_acc_cifar10 = evaluate_model(model, trainloader_cifar10, device)
            test_acc_cifar10 = evaluate_model(model, testloader_cifar10, device)
            print(f'Model {epoch} epoch. Train acc {train_acc_cifar10:.3f} Test acc {test_acc_cifar10:.3f}')

            torch.save(model.state_dict(), savedir)

        train_loss = 0.0
        total_train = 0
        correct_train = 0

        model.train()
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            with torch.no_grad():
                target_outputs = target_model(inputs)
                if hard_label:
                    target_outputs = torch.argmax(target_outputs, dim=1)
                else:
                    target_outputs = F.softmax(target_outputs, dim=1)
            outputs = model(inputs)
            if not hard_label:
                outputs = F.log_softmax(outputs, dim=1)
            loss = criterion(outputs, target_outputs)

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_student_acc.append(correct_train / total_train)
        train_loss_history.append(train_loss / len(train_dataloader))

        if do_eval:
            if epoch % epoch_eval == 0:
                model.eval()
                test_loss = 0.0
                correct_test = 0
                total_test = 0
                with torch.no_grad():
                    for data in test_dataloader:
                        inputs, labels = data
                        inputs = inputs.float().to(device)
                        labels = labels.to(device)

                        target_outputs = target_model(inputs)
                        if hard_label:
                            target_outputs = torch.argmax(target_outputs, dim=1)
                        else:
                            target_outputs = F.softmax(target_outputs, dim=1)
                        outputs = model(inputs)
                        if not hard_label:
                            outputs = F.log_softmax(outputs, dim=1)
                        loss = criterion(outputs, target_outputs)

                        _, predicted = torch.max(outputs.data, 1)

                        correct_test += (predicted == labels).sum().item()
                        total_test += labels.size(0)
                        test_loss += loss
        test_student_acc.append(correct_test / total_test)
        test_loss_history.append(test_loss / len(test_dataloader))

        # current_mse = models_mse(model, target_model)

        print(f'Epoch {epoch + 1} Train loss {train_loss_history[-1]:.3f} Last Test Loss {test_loss_history[-1]:.3f} Train Student Acc {train_student_acc[-1]:.3f}')
    return train_loss_history, test_loss_history

def models_mse(model1:nn.Module, model2:nn.Module)->float:
    with torch.no_grad():
        mse = 0.0
        parameters2 = list(model2.parameters())
        for ind, parameter1 in enumerate(model1.parameters()):
            parameter2 = parameters2[ind]
            mse += torch.sum((parameter1 - parameter2)**2).item()
        mse /= len(parameters2)
        return mse

def get_dataset(dataset_name:str):
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
    if dataset_name=='cifar100':
        trainset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        return trainset, testset
    elif dataset_name=='cifar10':
        trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        return trainset, testset


trigger_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

def get_trigger_set(size=100, indices=None):
    if indices is None:
        dataset = MNIST(root='./data', train=True, download=True, transform=trigger_transforms)
        indices = torch.randperm(len(dataset))[:size]

        return Subset(dataset, indices), indices
    else:
        dataset = MNIST(root='./data', train=True, download=True, transform=trigger_transforms)
        return Subset(dataset, indices)