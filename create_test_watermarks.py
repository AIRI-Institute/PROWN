import torch
import argparse
import glob
import copy

from torch.utils.data import DataLoader
from datasets import get_dataset, get_bounds, get_num_classes
from global_options import _models
from watermark_dataset import FINAL_DATASET
from utils import evaluate_model
    
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', help="dataset name", type=str, choices=['imagenet', 'cifar10', 'cifar100', 'mnist', 'svhn'], default='cifar10')
    parser.add_argument('--batch', help="batch_size", type=int, default=64)
    parser.add_argument('--device', help="device", type=str, default='cuda:0')
    parser.add_argument('--seed', help="seed", type=int, default=0)
    parser.add_argument('--model_name', help = "target model architecture", choices=list(_models.keys()), default='resnet34')
    parser.add_argument('--model_path', help="path to saved model", type=str, default='./models/teacher_cifar10_resnet34/model_1')
    parser.add_argument('--student_name', help = "stolen model architecture", choices=list(_models.keys()), default='resnet34')
    parser.add_argument('--student_path', help="path to stolen models", type=str, default='./models/stealing_resnet34_cifar10_soft')
    parser.add_argument('--N', help="size of trigger set", type=int, default=100)
    parser.add_argument('--sigma1', help="sigma in", type=float, default=None)
    parser.add_argument('--sigma2', help="sigma out", type=float, default=None)
    parser.add_argument('--M', help="number of proxy models", type=int, default=32)
    parser.add_argument('--threshold', help="threshold for proxy models", type=float, default=None)
    parser.add_argument('--use_train', help="watermarks based on train?", type=bool, default=False)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    if args.use_train:
        args.train_dataset = get_dataset(args.dataset, 'train')
    args.test_dataset = get_dataset(args.dataset, 'test')
    args.num_classes = get_num_classes(args.dataset)
    args.bounds = get_bounds(args.dataset)
    
    model = _models[args.model_name](num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.to(args.device)
    model.eval()
    
    args.model = model
    
    adv_dataset = FINAL_DATASET(args)
    adv_loader = DataLoader(adv_dataset, shuffle=False, batch_size=256)
    
    ### Evaluate accuracy on stolen models ###
    print("Start evaluate stolen models")
    path_models = glob.glob(args.student_path + '/*')
    
    new_model = _models[args.student_name](num_classes=args.num_classes)
    
    I_mean = []
    for model_path in path_models:
        new_model.load_state_dict(torch.load(model_path, map_location=args.device))
        new_model.to(args.device)
        new_model.eval()
        
        I_mean.append(evaluate_model(new_model, adv_loader, args.device))
        print("Accuracy ", I_mean[-1])
        
    I_mean = torch.tensor(I_mean)
    print("======================")
    print(f"Mean acc {I_mean.mean().item():.3f} +- {I_mean.std().item():.3f}")
    print("======================")