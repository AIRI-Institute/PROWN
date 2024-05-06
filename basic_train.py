import torch
from datasets import get_dataset
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate_model
import os

def train_model(model, dataset, num_epochs, device, do_eval=False, epoch_eval=5,  opt=None, savedir='./model.pt', save_iter=10):

    print(f'Training base model policy on {dataset}')
    BATCH_SIZE=512
    # Datasets on which stealing will be performed
    trainset = get_dataset(dataset, split='train')
    testset = get_dataset(dataset, split='test')

    criterion = nn.CrossEntropyLoss()

    if opt is None:
        opt = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
            
    model = model.to(device)  

    train_loss_history = []
    test_loss_history = []

    train_student_acc = []
    test_student_acc = []

    train_dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    # Checking save dir and save name
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    save_name = 0
    while os.path.isfile(os.path.join(savedir, f'model_{save_name}')):
        save_name += 1

    for epoch in tqdm(range(num_epochs)):

        if epoch % save_iter == 0:
            
            torch.save(model.state_dict(), os.path.join(savedir, f'model_{save_name}'))
            print(f'Model saved, {dataset} performance:')
            print(f'Model {epoch} epoch. Train acc {evaluate_model(model, train_dataloader, device):.3f} \
                  Test acc {evaluate_model(model, test_dataloader, device):.3f}')

        train_loss = 0.0
        total_train = 0
        correct_train = 0

        model.train()
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

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

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, predicted = torch.max(outputs.data, 1)

                        correct_test += (predicted == labels).sum().item()
                        total_test += labels.size(0)
                        test_loss += loss.item()
        test_student_acc.append(correct_test / total_test)
        test_loss_history.append(test_loss / len(test_dataloader))

        print(f'Epoch {epoch + 1} Train loss {train_loss_history[-1]:.3f} Last Test Loss {test_loss_history[-1]:.3f} Train Student Acc {train_student_acc[-1]:.3f}')
    return train_loss_history, test_loss_history
