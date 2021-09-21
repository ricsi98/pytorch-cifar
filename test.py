from loader import CV5CIFAR10
import matplotlib.pyplot as plt

for i in range(5):
    ds = CV5CIFAR10("./data", train=False, download=False, current_fold=i)
    img = ds[0][0]
    img.show()