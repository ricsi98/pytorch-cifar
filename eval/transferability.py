import argparse

from utils import load_model, PREPROCESS
from adversarial import fgsm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


LOGS = False
IS_2N = False


def transferability(model, adv_model, epsilon):
    test_set = CIFAR10(root='./data', train=False, download=True, transform=PREPROCESS)
    loader = DataLoader(test_set, batch_size=64, shuffle=False)

    all_attacks = 0
    successful_attacks = 0
    n_batches = test_set.data.shape[0] // loader.batch_size + 1

    for batch_idx, (inputs, targets) in enumerate(loader):
        x_ = fgsm(inputs, targets, adv_model, epsilon, nn.CrossEntropyLoss())

        y_adv = torch.argmax(adv_model(x_), dim=1)
        y_mdl = torch.argmax(model(x_), dim=1)

        if IS_2N:
            y_mdl = torch.remainder(y_mdl, 10)

        successful_adv = (y_adv != targets).double()
        successful_mdl = (y_mdl != targets).double()

        successful_transfered = torch.mul(successful_mdl, successful_adv)
        n_successful_attack = torch.sum(successful_transfered).item()
        n_attack = torch.sum(successful_adv).item()

        successful_attacks += n_successful_attack
        all_attacks += n_attack

        if LOGS:
            print(f"Batch {batch_idx}/{n_batches}, current transferability: {'?' if all_attacks == 0 else round(successful_attacks / all_attacks, 5)}")

    if all_attacks == 0:
        raise Exception("No attacks were successful")
    return successful_attacks / all_attacks


def main():
    parser = argparse.ArgumentParser(description="Calculate transferability score")
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--nClasses", type=int, help="model number of classes, either 10, 11 or 20")
    parser.add_argument("--advModel", type=str, help="adversary model path")
    parser.add_argument("--epsilon", type=float, help="fgsm epsilon")
    parser.add_argument("--verbose", "-v", action='store_true')
    args = parser.parse_args()

    model = load_model(args.model, args.nClasses)
    adv_model = load_model(args.advModel, 10)

    global LOGS
    LOGS = args.verbose
    global IS_2N
    IS_2N = args.nClasses == 20

    print("TRANSFERABILITY", transferability(model, adv_model, args.epsilon))


if __name__ == '__main__':
    main()
