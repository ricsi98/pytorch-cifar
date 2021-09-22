from collections import OrderedDict

import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

from utils import PREPROCESS, get_model
import torch
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    # load model
    model = get_model('vgg19')
    data = torch.load("./checkpoint/ckpt.pth")

    renamedData = OrderedDict()
    for key in data['net'].keys():
        key_ = key[7:]
        renamedData[key_] = data['net'][key]

    model.load_state_dict(renamedData)

    # load test data
    testSet = CIFAR10(root='./data', train=False, download=True, transform=PREPROCESS)
    loader = torch.utils.data.DataLoader(testSet, batch_size=128, shuffle=False)
    # apply model

    import numpy as np
    y = []
    y_ = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            output = model(inputs)
            y = y + [targets.numpy()]
            y_ = y_ + [np.argmax(output.numpy(), axis=1)]

    y = np.concatenate(y)
    y_ = np.concatenate(y_)
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
