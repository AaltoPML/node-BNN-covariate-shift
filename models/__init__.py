from .utils import *
from .vgg import *
from .sam import *
from .resnet import *
from .all_conv import *
from .preactresnet import *

def get_model_from_config(config, dropout=True):
    model_name = config.get('model_name', config.get('model'))
    if model_name ==  'StoVGG16':
        return StoVGG16(config['num_classes'], config['n_components'], 1.0, 0.3, (1.0, 0.5), (0.05, 0.02), mode=config.get('noise_mode', 'in'))
    if model_name ==  'StoVGG16BN':
        return StoVGG16BN(config['num_classes'], config['n_components'], 1.0, 0.3, (1.0, 0.5), (0.05, 0.02))
    if model_name ==  'StoVGG16CBN':
        return StoVGG16CBN(config['num_classes'], config['n_components'], 1.0, 0.3, (1.0, 0.5), (0.05, 0.02))
    if model_name ==  'StoResNet18':
        return StoResNet18(config['num_classes'], config['n_components'], 1.0, 0.3, (1.0, 0.5), (0.05, 0.02), mode=config.get('noise_mode', 'in'))
    if model_name ==  'StoPreActResNet18':
        return StoPreActResNet18(config['num_classes'], config['n_components'], 1.0, 0.3, (1.0, 0.5), (0.05, 0.02), mode=config.get('noise_mode', 'in'))
    if model_name ==  'DetResNet18':
        return DetResNet18(config['num_classes'])
    if model_name ==  'DetVGG16':
        return DetVGG16(config['num_classes'], dropout)
    if model_name == 'StoAllConv':
        return StoAllConv(config['num_classes'], config['n_components'], 1.0, 0.3, (1.0, 0.5), (0.05, 0.02), mode=config.get('noise_mode', 'in'))
    if model_name == 'DetAllConv':
        return DetAllConv(config['num_classes'])
    if model_name == 'DetPreActResNet18':
        return DetPreActResNet18(config['num_classes'])