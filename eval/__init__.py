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
        self.evaluator = None

        self.needs_plain_output = True
        self.needs_plain_adv_output = True
        self.needs_attack_output = True
        self.needs_attack_adv_output = True

    def register_evaluator(self, evaluator):
        self.evaluator = evaluator

    @abstractmethod
    def process(self, mdl_output, adv_output, plain_output, plain_adv_output, target):
        raise NotImplementedError()

    @abstractmethod
    def result(self):
        raise NotImplementedError()

    @abstractmethod
    def peek_result(self):
        raise NotImplementedError()

    def _probs_to_labels(self, mdl_output):
        if self.evaluator.model_n_classes == 11:
            # NULL
            return torch.argmax(mdl_output[:, :-1], dim=1)
        y = torch.argmax(mdl_output, dim=1)
        if self.evaluator.model_n_classes == 10:
            # reference
            return y
        # 2N
        return torch.remainder(y, 10)


class Evaluator:

    def __init__(self, adv_model_path: str, batch_size: int = 64, verbose: bool = False):
        self.dataset = CIFAR10(root='./data', train=False, download=True, transform=PREPROCESS)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

        self.n_batches = self.dataset.data.shape[0] // batch_size + 1

        self.model = None
        self.model_n_classes = None
        self.adv_model = load_model(adv_model_path, 10)

        self._verbose = verbose

    def load_model(self, path: str, n_classes: int):
        self.model_n_classes = n_classes
        self.model = load_model(path, n_classes)

    def evaluate(self, evaluation_function: EvaluationFunction, epsilon: float):
        assert self.model is not None, "you should call load_model first!"

        evaluation_function.register_evaluator(self)

        for batch_idx, (inputs, targets) in enumerate(self.loader):
            x_ = fgsm(inputs, targets, self.adv_model, epsilon, nn.CrossEntropyLoss())

            pre_adv_output = None
            post_adv_output = None
            pre_mdl_output = None
            post_mdl_output = None

            with torch.no_grad():
                if evaluation_function.needs_plain_output:
                    pre_mdl_output = self.model(inputs)
                if evaluation_function.needs_attack_output:
                    post_mdl_output = self.model(x_)
                if evaluation_function.needs_plain_adv_output:
                    pre_adv_output = self.adv_model(inputs)
                if evaluation_function.needs_attack_adv_output:
                    post_adv_output = self.adv_model(x_)

            evaluation_function.process(post_mdl_output, post_adv_output,
                                        pre_mdl_output, pre_adv_output, targets)

            if self._verbose:
                print(
                    f"""Batch {batch_idx}/{self.n_batches}, \
                        current {evaluation_function.name}: {evaluation_function.peek_result()}""")

        return evaluation_function.result()
