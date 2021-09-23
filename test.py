import torch
from loader import CV5CIFAR10
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from adversarial import AdversarialTransform
from utils import TransformParameterWrapper, TransformParameterKeepFirst


tl = [
    TransformParameterWrapper(transforms.ToTensor()),
    TransformParameterWrapper(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))),
    AdversarialTransform(100, "./checkpoint/reference.pth", "vgg19", 0.7, "normal"),
    TransformParameterKeepFirst()
]
transform = torchvision.transforms.Compose(tl)
test_set = CV5CIFAR10(root='./data', train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False)

for i, (X, y) in enumerate(loader):
    img_grid = torchvision.utils.make_grid(X, nrow=5)
    plt.imshow(img_grid.permute(1,2,0))
    plt.show()
    break