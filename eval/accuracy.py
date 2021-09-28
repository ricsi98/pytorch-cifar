import torch
import argparse

from . import Evaluator, EvaluationFunction


LOGS = False
IS_2N = False


class Accuracy(EvaluationFunction):

    def __init__(self):
        super().__init__()
        self.name = "Accuracy"
        self.correct = 0
        self.all = 0

        self.needs_plain_output = False
        self.needs_plain_adv_output = False

    def process(self, mdl_output, adv_output, plain_output, plain_adv_output, target):
        y = self._probs_to_labels(mdl_output)
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
