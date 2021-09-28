'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import torch
import warnings

import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms

from models.vgg import VGG
from collections import OrderedDict


def get_model(name: str, n_classes:int=10):
    if name == "vgg11":
        return VGG('VGG11', n_classes)
    elif name == "vgg16":
        return VGG('VGG16', n_classes)
    elif name == "vgg19":
        return VGG('VGG19', n_classes)


def match_state_dict_keys(state_dict: OrderedDict):
    if "module" in list(state_dict.keys())[0][:8]:
        renamed_data = OrderedDict()
        for key in state_dict.keys():
            key_ = key[7:]
            renamed_data[key_] = state_dict[key]
        return renamed_data
    return state_dict


class TransformParameterWrapper(nn.Module):
    """Can be used to pass further information which the current transform does not need"""
    def __init__(self, transform):
        super(TransformParameterWrapper, self).__init__()
        self._transform = transform

    def forward(self, x):
        return (self._transform(x[0]), *x[1:])


class TransformParameterKeepFirst(nn.Module):

    def __init__(self):
        super(TransformParameterKeepFirst, self).__init__()
    
    def forward(self, x):
        return x[0]


PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

if sys.platform in ['linux1', 'linux2']:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
else:
    term_width = 10


#_, term_width = os.popen('stty size', 'r').read().split()
#erm_width = 60 #int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def load_model(path: str, n_classes: int):
    assert n_classes in [10, 11, 20], "nClasses must be either 10, 11 or 20"
    model = get_model('vgg19', n_classes)
    data = torch.load(path)

    try:
        model.load_state_dict(data['net'])
    except RuntimeError:
        warnings.warn("Pytorch model state_dict key mismatch!")
        if "module" in list(data['net'].keys())[0][:8]:
            renamed_data = OrderedDict()
            for key in data['net'].keys():
                key_ = key[7:]
                renamed_data[key_] = data['net'][key]
            state_dict = renamed_data
        else:
            state_dict = data

        model.load_state_dict(state_dict)

    return model
