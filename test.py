import torch
from loader import CV5CIFAR10
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from adversarial import AdversarialTransform, AdversarialLabelTransform
from utils import TransformParameterWrapper, TransformParameterKeepFirst, load_model
import numpy as np
from sklearn.metrics import accuracy_score
from eval.accuracy import Accuracy
from eval import Evaluator


RATE = 0.5

tl = [
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
]

transform = torchvision.transforms.Compose(tl)

test_set = CV5CIFAR10(root='./data', train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(test_set, batch_size=25, shuffle=False)

evaluator = Evaluator("./checkpoint/reference.pth", 25, True)
evaluator.load_model("./checkpoint/base0.pth", 10)
print("ACC", evaluator.evaluate(Accuracy(), 0))

"""model = load_model("./checkpoint/base0.pth", 10)

model.eval()
class MockEval:
    def __init__(self):
        self.model_n_classes = 10

acc = Accuracy()
acc.register_evaluator(MockEval())

for input, target in loader:
    output = model(input)
    acc.process(output, None, None, None, target)
    print("ACC", acc.peek_result())
print("ACC", acc.result())"""


"""y, y_ = [], []
for input, target in loader:
    print("asd")
    output = model(input)
    labels = torch.argmax(output, dim=1)
    y += target.numpy().tolist()
    y_ += labels.detach().numpy().tolist()"""

y, y_ = np.array(y), np.array(y_)
print("ACC", accuracy_score(y, y_))