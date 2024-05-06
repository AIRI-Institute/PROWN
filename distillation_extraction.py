import os
import argparse
import torch
import torch.nn.functional as F

from datasets import get_dataset, get_num_classes
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate_model
from global_options import _models

def distillation_loss(preds, labels, teacher_preds, T, alpha):
    teacher_loss = F.kl_div(F.log_softmax(preds / T, dim=-1), F.softmax(teacher_preds / T, dim=-1), reduction='batchmean')
    ground_loss = F.cross_entropy(preds, labels)

    return (1 - alpha) * ground_loss + alpha * teacher_loss

def steal(model, teacher_model, dataset, evalset, num_epochs, device, do_eval=False, epoch_eval=5,  opt=None, lr_scheduler=None, savedir='./model.pt', save_iter=10, policy='soft', T=20.0, alpha=0.7, stop_acc=0.99):

    print(f'Training with policy - {policy}. Training on {dataset}. Evaluation on {evalset}')
    BATCH_SIZE=256
    # Datasets on which stealing will be performed
    trainset = get_dataset(dataset, split='train')
    testset = get_dataset(dataset, split='test')

    # Datasets on which teacher model was trained
    # On this datasets evaluation of student model will be performed
    eval_trainset = get_dataset(evalset, split='train')
    eval_testset = get_dataset(evalset, split='test')

    # KL divergence with soft labels
    if policy=='soft':
        # Loss is not being computed for labels
        criterion = lambda preds_, labels_, teacher_preds_: distillation_loss(preds_, labels_, teacher_preds_, T, alpha=1)
    # Cross entropy loss for hard labeled output of teacher model
    elif policy=='hard':
        criterion = lambda preds_, labels_, teacher_preds_: nn.CrossEntropyLoss()(preds_, torch.argmax(teacher_preds_, dim=1))
    # Regularization with ground truth labels
    elif policy=='rgt':
        criterion = lambda preds_, labels_, teacher_preds_: distillation_loss(preds_, labels_, teacher_preds_, T, alpha=alpha)
    else:
        raise ValueError(f'Invalid value: {policy}. Expected one of: [soft, hard, rgt]')

    if opt is None:
        opt = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)

    if lr_scheduler is None:
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.3   )
            
    model = model.to(device)  
    teacher_model = teacher_model.to(device)

    train_loss_history = []
    test_loss_history = []

    train_student_acc = []
    test_student_acc = []

    train_dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    eval_train_dataloader = DataLoader(eval_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    eval_test_dataloader = DataLoader(eval_testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    # Evaluate Teacher model
    train_teacher_acc = evaluate_model(teacher_model, eval_train_dataloader, device)
    test_teacher_acc = evaluate_model(teacher_model, eval_test_dataloader, device)
    print(f'Evaluated Teacher Model on {evalset}')
    print(f'Teacher Model Train Acc {train_teacher_acc:.3f}, Test Acc {test_teacher_acc:.3f}')

    # Checking save dir and save name
    os.makedirs(savedir, exist_ok=True)
    
    save_name = 0
    while os.path.isfile(os.path.join(savedir, f'model_{save_name}')):
        save_name += 1

    best_test_acc = 0

    for epoch in tqdm(range(num_epochs)):

        train_loss = 0.0
        total_train = 0
        correct_train = 0

        model.train()
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            outputs = model(inputs)

            loss = criterion(outputs, labels, teacher_outputs)

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_student_acc.append(correct_train / total_train)
        train_loss_history.append(train_loss / len(train_dataloader))

        lr_scheduler.step()

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

                        teacher_outputs = teacher_model(inputs)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels, teacher_outputs)

                        _, predicted = torch.max(outputs.data, 1)

                        correct_test += (predicted == labels).sum().item()
                        total_test += labels.size(0)
                        test_loss += loss.item()
        test_student_acc.append(correct_test / total_test)
        test_loss_history.append(test_loss / len(test_dataloader))

        if epoch % save_iter == 0:
            
            
            train_acc = evaluate_model(model, eval_train_dataloader, device)
            test_acc = evaluate_model(model, eval_test_dataloader, device)
            
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), os.path.join(savedir, f'model_{save_name}'))
                print(f'Model saved, {evalset} performance:')
                print(f'Model {epoch} epoch. Train acc {train_acc:.3f} \
                  Test acc {test_acc:.3f}')
            else:
                print(f'Model not saved, {evalset} perfomance:')
                print(f'Model {epoch} epoch. Train acc {train_acc:.3f} \
                  Test acc {test_acc:.3f}')
                print(f'Best test acc: {best_test_acc}')
            if test_acc >= stop_acc:
                return train_loss_history, test_loss_history

        print(f'Epoch {epoch + 1} Train loss {train_loss_history[-1]:.4f} Last Test Loss {test_loss_history[-1]:.4f} Train Student Acc {train_student_acc[-1]:.4f}')
    return train_loss_history, test_loss_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', help="dataset name", type=str, choices=['imagenet', 'cifar10', 'cifar100', 'mnist', 'svhn'], default='cifar10')
    parser.add_argument('--evalset', help="evalset name", type=str, choices=['imagenet', 'cifar10', 'cifar100', 'mnist', 'svhn'], default='cifar10')
    parser.add_argument('--batch', help="batch_size", type=int, default=64)
    parser.add_argument('--device', help="device", type=str, default='cuda:0')
    parser.add_argument('--seed', help="seed", type=int, default=0)
    parser.add_argument('--model_name', help = "target model architecture", choices=list(_models.keys()), default='resnet34')
    parser.add_argument('--model_path', help="path to saved model", type=str, default='./models/teacher_cifar10_resnet34/model_1')
    parser.add_argument('--student_name', help = "stolen model architecture", choices=list(_models.keys()), default='resnet34')
    parser.add_argument('--save_dir', help="Path to save model", type=str, default=None)
    parser.add_argument('--num_models', help="Amount of models to be stealed", type=int, default=10)
    parser.add_argument('--policy', help="device", type=str, choices=['soft', 'hard', 'rgt'], default='soft')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    args.num_classes = get_num_classes(args.dataset)
    teacher_model = _models[args.model_name](num_classes=args.num_classes)
    teacher_model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    if args.save_dir is None:
        args.save_dir = './models/stealing_' + args.student_name + '_' + args.evalset + '_' + args.policy
        
    for i in range(args.num_models):
        student_model = _models[args.student_name](num_classes=args.num_classes)
        steal(student_model, teacher_model, dataset=args.dataset, evalset=args.evalset,
            num_epochs=100, device=args.device, do_eval=True, epoch_eval=5, 
            savedir=args.save_dir, save_iter=5, T=1, alpha=0.3, policy=args.policy, stop_acc=0.99)