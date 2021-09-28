import torch
import argparse

from . import Evaluator, EvaluationFunction


def successful_attacks(pre_attack, post_attack, target):
    pre_correct = (pre_attack == target).double()
    post_incorrect = (post_attack != target).double()
    return torch.mul(pre_correct, post_incorrect)


class Transferability(EvaluationFunction):

    def __init__(self):
        super().__init__()
        self.name = "Transferability"
        self.successful = 0
        self.adv_successful = 0

    def process(self, mdl_output, adv_output, plain_output, plain_adv_output, target):
        with torch.no_grad():
            y_mdl_plain = self._probs_to_labels(plain_output)
            y_mdl_attack = self._probs_to_labels(mdl_output)
            y_adv_plain = torch.argmax(plain_adv_output, dim=1)
            y_adv_attack = torch.argmax(adv_output, dim=1)

            mdl_successful = successful_attacks(y_mdl_plain, y_mdl_attack, target)
            adv_successful = successful_attacks(y_adv_plain, y_adv_attack, target)

            n_transferred = torch.sum(torch.mul(mdl_successful, adv_successful)).item()

            self.successful += int(n_transferred)
            self.adv_successful += int(torch.sum(adv_successful).item())

    def result(self):
        if self.adv_successful == 0:
            return 0
        return self.successful / self.adv_successful

    def peek_result(self):
        return self.result()


LOGS = False
IS_2N = False
DEVICE = "cpu"
BS = 64


def transferability(model_path, n_classes, adv_model_path, epsilon):
    evaluator = Evaluator(adv_model_path, verbose=LOGS, device=DEVICE, batch_size=BS)
    evaluator.load_model(model_path, n_classes)
    return evaluator.evaluate(Transferability(), epsilon)


def main():
    parser = argparse.ArgumentParser(description="Calculate transferability score")
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

    print("TRANSFERABILITY", transferability(args.model, args.nClasses, args.advModel, args.epsilon))


if __name__ == '__main__':
    main()
