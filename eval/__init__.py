import torch
import torch.nn as nn

from adversarial import fgsm
from utils import load_model, PREPROCESS

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from abc import abstractmethod


class EvaluationFunction:

    def __init__(self):
        self.name = "NotImplemented"

    @abstractmethod
    def process(self, mdl_output, adv_output, target):
        raise NotImplementedError()

    @abstractmethod
    def result(self):
        raise NotImplementedError()

    @abstractmethod
    def peek_result(self):
        raise NotImplementedError()


class Evaluator:

    def __init__(self, adv_model_path: str, batch_size: int = 64, verbose: bool = False):
        self.dataset = CIFAR10(root='./data', train=False, download=True, transform=PREPROCESS)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

        self.n_batches = self.dataset.data.shape[0] // batch_size + 1

        self.model = None
        self.adv_model = load_model(adv_model_path, 10)

        self._verbose = verbose

    def load_model(self, path: str, n_classes: int):
        self.model = load_model(path, n_classes)

    def evaluate(self, evaluation_function: EvaluationFunction, epsilon: float):
        assert self.model is not None, "you should call load_model first!"

        for batch_idx, (inputs, targets) in enumerate(self.loader):
            x_ = fgsm(inputs, targets, self.adv_model, epsilon, nn.CrossEntropyLoss())

            adv_output = self.adv_model(x_)
            mdl_output = self.model(x_)

            evaluation_function.process(mdl_output, adv_output, targets)

            if self._verbose:
                print(
                    f"""Batch {batch_idx}/{self.n_batches}, \
                        current {evaluation_function.name}: {evaluation_function.peek_result()}""")

        return evaluation_function.result()
