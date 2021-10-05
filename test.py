from eval import Evaluator
import sys
from eval.accuracy import *
from eval.confmat import ConfMat2n


if __name__ == '__main__':
    e = Evaluator("./checkpoint/reference.pth", verbose=False)
    e.load_model("./checkpoint/2n/2n1.pth", 20)

    metric = ConfMat2n()#BinaryAccuracy(simple_2n_classifier)
    epsilon = float(sys.argv[-1])
    print("EPSILON=",epsilon)
    print(e.evaluate(metric, epsilon))