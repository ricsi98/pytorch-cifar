from . import EvaluationFunction, Evaluator
import torch
import argparse
from sklearn.metrics import roc_auc_score

class AUC(EvaluationFunction):
    
    def __init__(self):
        super().__init__()
        self.name = "ROC-AUC"
        self.needs_attack_adv_output = False
        self.needs_plain_adv_output = False
        self.y = []
        self.y_ = []

    def _get_binary_labels(self, output):
        n_classes = self.evaluator.model_n_classes

        if n_classes == 11:
            return torch.ones_like(output[:,0]) - output[:, -1]
        elif n_classes == 20:
            return torch.sum(output[:, :10], dim=1)
        raise NotImplemented("Not implemented for other than {11,20} labels")

    def process(self, mdl_output, adv_output, plain_output, plain_adv_output, target, epsilon):
        pre_binary = self._get_binary_labels(mdl_output)
        self.y_ += pre_binary.detach().cpu().numpy().tolist()
        zeros = torch.zeros_like(pre_binary)
        self.y += zeros.detach().cpu().numpy().tolist()
        if epsilon == 0:
            return
        ones = torch.ones_like(pre_binary)
        self.y += ones.detach().cpu().numpy().tolist()
        post_binary = self._get_binary_labels(plain_output)
        self.y_ += post_binary.detach().cpu().numpy().tolist()
        


    def result(self):
        if len(self.y) > 0 and sum(self.y) != len(self.y):
            return roc_auc_score(self.y, self.y_)
        return 0

    def peek_result(self):
        return self.result()


def auc(model_path, n_classes, adv_model, epsilon):
    evaluator = Evaluator(adv_model, verbose=True, device="cpu", batch_size=25)
    evaluator.load_model(model_path, n_classes)
    return evaluator.evaluate(AUC(), epsilon)



def main():
    parser = argparse.ArgumentParser(description="Calculate ROC-AUC score")
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--nClasses", type=int, help="model number of classes, either 10, 11 or 20")
    parser.add_argument("--advModel", type=str, help="adversary model path")
    parser.add_argument("--epsilon", type=float, help="fgsm epsilon")
    parser.add_argument("--verbose", "-v", action='store_true')
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--batchSize", type=int, default=64)
    args = parser.parse_args()

    """global LOGS
    LOGS = args.verbose
    global IS_2N
    IS_2N = args.nClasses == 20
    global DEVICE
    DEVICE = "cuda" if (torch.cuda.is_available() and args.gpu) else "cpu"
    print("DEVICE", DEVICE)
    global BS
    BS = args.batchSize
    print("BATCH SIZE", BS)"""

    print("TRANSFERABILITY", auc(args.model, args.nClasses, args.advModel, args.epsilon))


if __name__ == '__main__':
    main()
