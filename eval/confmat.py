import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from utils import PREPROCESS, get_model
from adversarial import fgsm

from torchvision.datasets import CIFAR10
from sklearn.metrics import confusion_matrix

import torch.nn as nn


DEBUG = False
USE_ADV = False


def load_model(path: str, n_classes: int):
    assert n_classes in [10, 11, 20], "nClasses must be either 10, 11 or 20"
    model = get_model('vgg19', n_classes)
    data = torch.load(path)

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


def eval_model(model, loader, adv_model, epsilon):
    y = []
    y_ = []
    if epsilon > 0 and adv_model:
        for batch_idx, (inputs, targets) in enumerate(loader):
            x_ = fgsm(inputs, targets, adv_model, epsilon, nn.CrossEntropyLoss())
            output = model(x_)
            y = y + [targets.detach().numpy()]
            y_ = y_ + [np.argmax(output.detach().numpy(), axis=1)]

            if DEBUG and batch_idx > 4:
                break
    else:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                output = model(inputs)
                y = y + [targets.numpy()]
                y_ = y_ + [np.argmax(output.numpy(), axis=1)]

                if DEBUG and batch_idx > 4:
                    break

    y = np.concatenate(y)
    y_ = np.concatenate(y_)
    return y, y_


def main():
    parser = argparse.ArgumentParser(description='Generate confusion matrix for model')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    parser.add_argument('--model', type=str,
                        help='Path to model checkpoint (Currently VGG19 is the only supported model)')
    parser.add_argument('--advModel', type=str, default="",
                        help='Path to adversary model checkpoint (Currently VGG19 only)')
    parser.add_argument('--epsilon', type=float, default=.0, help='FGSM epsilon')
    parser.add_argument('--nClasses', type=int, default=10, help='Number of classes, either 10, 11 or 20')

    args = parser.parse_args()

    global DEBUG
    global USE_ADV
    DEBUG = args.debug
    USE_ADV = args.advModel and args.epsilon > 0
    if DEBUG:
        print("DEBUG MODE")

    model = load_model(args.model, args.nClasses)
    adv_model = load_model(args.advModel, 10) if USE_ADV else None

    # load test data
    test_set = CIFAR10(root='./data', train=False, download=True, transform=PREPROCESS)
    loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    y, y_ = eval_model(model, loader, adv_model, args.epsilon)
    cnf = confusion_matrix(y, y_)
    print(cnf)

    # plot
    fig, ax = plt.subplots()
    for i in range(10):
        for j in range(10):
            c = cnf[j, i]
            ax.text(i, j, str(c), va='center', ha='center', c='white' if i != j else 'black')
    ax.matshow(cnf)

    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    ax.set_xticks([i for i in range(10)])
    ax.set_yticks([i for i in range(10)])

    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground truth')

    plt.show()


if __name__ == '__main__':
    main()
