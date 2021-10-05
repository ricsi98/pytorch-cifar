import torch
import argparse

from . import Evaluator, EvaluationFunction


LOGS = False
IS_2N = False
DEVICE = "cpu"
BS = 64


class Accuracy(EvaluationFunction):

    def __init__(self):
        super().__init__()
        self.name = "Accuracy"
        self.correct = 0
        self.all = 0

        self.needs_plain_output = False
        self.needs_plain_adv_output = False

    def process(self, mdl_output, adv_output, plain_output, plain_adv_output, target, epsilon):
        with torch.no_grad():
            y = self._probs_to_labels(mdl_output)
            n_correct = torch.sum((y == target).double()).item()
            self.correct += int(n_correct)
            self.all += target.shape[0]

    def result(self):
        return self.correct / self.all

    def peek_result(self):
        return self.result()


def simple_2n_classifier(y):
    print(y[0], torch.softmax(y[0], dim=0))
    return torch.sum(torch.softmax(y[:, 10:], dim=1), dim=1)


class BinaryAccuracy(Accuracy):

    def __init__(self, classification_function):
        super().__init__()
        self.name = "Binary Accuracy"
        self.needs_plain_adv_output = False
        self.needs_attack_adv_output = False
        self.needs_plain_output = True
        self.clf = classification_function

    def process(self, mdl_output, adv_output, plain_output, plain_adv_output, target, epsilon):
        with torch.no_grad():
            post_y = self.clf(mdl_output)
            pre_y = self.clf(plain_output)
            probs = torch.cat((post_y, pre_y), dim=0)
            y_ = torch.round(probs)

            ones = torch.ones_like(post_y)
            zeros = torch.zeros_like(pre_y)
            labels = torch.cat((ones, zeros), dim=0)

            self.all += labels.shape[0]
            print(probs)
            self.correct += (y_ == labels).int().sum().item()
            print(self.all, self.correct)



def accuracy(model_path, n_classes, adv_model_path, epsilon):
    evaluator = Evaluator(adv_model_path, verbose=LOGS, device=DEVICE, batch_size=BS)
    evaluator.load_model(model_path, n_classes)
    return evaluator.evaluate(Accuracy(), epsilon)


def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy score")
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--nClasses", type=int, help="model number of classes, either 10, 11 or 20")
    parser.add_argument("--advModel", type=str, help="adversary model path")
    parser.add_argument("--epsilon", type=float, help="fgsm epsilon")
    parser.add_argument("--verbose", "-v", action='store_true')
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--batchSize", type=int, default=64)
    args = parser.parse_args()

    global LOGS
    LOGS = args.verbose
    global IS_2N
    IS_2N = args.nClasses == 20
    global DEVICE
    DEVICE = "cuda" if (torch.cuda.is_available() and args.gpu) else "cpu"
    print("DEVICE", DEVICE)
    global BS
    BS = args.batchSize
    print("BATCH SIZE", BS)

    print("ACCURACY", accuracy(args.model, args.nClasses, args.advModel, args.epsilon))


if __name__ == '__main__':
    main()
