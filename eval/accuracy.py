import argparse

import torch
import torch.nn as nn

from . import Evaluator, EvaluationFunction

from utils import load_model, PREPROCESS
from adversarial import fgsm

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


LOGS = False
IS_2N = False


class Accuracy(EvaluationFunction):

    def __init__(self):
        self.name = "Accuracy"
        self.correct = 0
        self.all = 0

    def process(self, mdl_output, adv_output, target):
        y = torch.argmax(mdl_output, dim=1)
        n_correct = torch.sum((y == target).double()).item()
        self.correct += n_correct
        self.all += target.shape[0]

    def result(self):
        return self.correct / self.all

    def peek_result(self):
        return self.result()


def accuracy(model_path, n_classes, adv_model_path, epsilon):
    evaluator = Evaluator(adv_model_path, verbose=LOGS)
    evaluator.load_model(model_path, n_classes)
    return evaluator.evaluate(Accuracy(), epsilon)




def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy score")
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--nClasses", type=int, help="model number of classes, either 10, 11 or 20")
    parser.add_argument("--advModel", type=str, help="adversary model path")
    parser.add_argument("--epsilon", type=float, help="fgsm epsilon")
    parser.add_argument("--verbose", "-v", action='store_true')
    args = parser.parse_args()

    global LOGS
    LOGS = args.verbose
    global IS_2N
    IS_2N = args.nClasses == 20

    print("ACCURACY", accuracy(args.model, args.nClasses, args.advModel, args.epsilon))


if __name__ == '__main__':
    main()
