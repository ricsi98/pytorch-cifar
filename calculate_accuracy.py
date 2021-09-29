from eval.accuracy import Accuracy
from eval import Evaluator
from utils import load_model
import os
import numpy as np

def get_n_classes(dir):
    if dir == "2n": return 20
    elif dir == "null": return 11
    return 10

DIRS = ["2n", "null", "base"]

evaluator = Evaluator("./checkpoint/reference.pth", 512, True, "cuda")

OUTPUT = {
    "epsilon": np.linspace(0,1,20)
}

for eps in OUTPUT['epsilon']:
    for dir in DIRS:
        for model in os.listdir("./checkpoint/" + dir):
            print("MODEL", model)
            path = "./checkpoint/" + dir + "/" + model
            name = model.split(".")[0]
            evaluator.load_model(path, get_n_classes(dir))
            acc = evaluator.evaluate(Accuracy(), eps)
            if name not in OUTPUT.keys():
                OUTPUT[name] = []
            OUTPUT[name] += [acc]

import pickle

with open("accuracy.pickle", "wb") as f:
    pickle.dump(OUTPUT, f)
