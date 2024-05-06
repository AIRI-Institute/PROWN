from torch.optim import SGD, Adam
from torchvision.models import vgg11
from models import resnet18, resnet34, resnet50

_optimizers = {
        'SGD': SGD, 
          'Adam':Adam
          }

_models = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'vgg11': vgg11
    }