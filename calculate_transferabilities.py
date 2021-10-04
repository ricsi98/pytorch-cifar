import os
import json
import numpy as np

from eval import Evaluator
from eval.transferability import Transferability
from utils import load_model

def get_n_classes(dir):
    if dir == "2n": return 20
    elif dir == "null": return 11
    return 10

DIRS = ["2n", "null", "base"]
epsilons = np.linspace(0, 1, 20)

OUTPUT = {}

adv_model = load_model("./checkpoint/reference.pth", 10)

evaluator = Evaluator("./checkpoint/reference.pth", batch_size=1000, device="cuda")

for e in epsilons:
    print("EPSILON =", e)
    OUTPUT[e] = {}
    for d in DIRS:
        checkpoints = os.listdir("./checkpoint/" + d)
        for cp_name in checkpoints:
            model_path = "./checkpoint/" + d + "/" + cp_name
            evaluator.load_model(model_path, get_n_classes(d))
            tf = evaluator.evaluate(Transferability(), e)
            model_name = model_path.split("/")[-1].split(".")[0]
            print(model_name, tf)
            OUTPUT[e][model_name] = tf

with open("results.pickle", "w") as f:
    json.dump(OUTPUT, f)
