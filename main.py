'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from loader import CV5CIFAR10
from adversarial import AdversarialTransform, AdversarialLabelTransform
from utils import progress_bar, get_model, TransformParameterWrapper, TransformParameterKeepFirst


def get_mode(n_classes: int):
    assert n_classes in [10, 11, 20]
    if n_classes == 10:
        return "normal"
    elif n_classes == 11:
        return "null"
    elif n_classes == 20:
        return "2n"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--fold', default=-1, type=int, help='current fold [-1,5). -1 means no CV')
    parser.add_argument('--model', default='vgg19', type=str, help='model to use')
    parser.add_argument('--nClasses', default=10, type=int, help='Number of classes to use')
    parser.add_argument('--adv', default="", type=str,
                        help='Path to adversarial model checkpoint, if not given, no adv training will be performed')
    parser.add_argument('--epsilon', default=0, type=float,
                        help='FGSM epsilon')
    parser.add_argument('--advRatio', default=0.5, type=float,
                        help='Ratio of adversarial examples during training')
    parser.add_argument('--noGpu', action='store_true', help='Force use cpu')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.noGpu:
        print("FORCE USING CPU")
        device = "cpu"

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    
    transform_train = transforms.Compose(transform_train_list)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    print("USING FOLD = ", args.fold)

    trainset = CV5CIFAR10(root='./data', current_fold=args.fold, train=True, download=True,
                          transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=25, shuffle=True, num_workers=2)

    testset = CV5CIFAR10(root='./data', current_fold=args.fold, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..', args.model)
    net = get_model(args.model, args.nClasses)
    net = net.to(device)

    if device == 'cuda':
        print("Using parallel")
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    print("USING DEVICE", "cpu" if not torch.cuda.is_available() else torch.cuda.current_device())

    ADV_TRAINING = args.adv != ""

    if ADV_TRAINING:
        adv_transform_inputs = AdversarialTransform(args.epsilon, args.adv, "vgg19",
                                                    args.advRatio, get_mode(args.nClasses), device)
        adv_transform_labels = AdversarialLabelTransform(get_mode(args.nClasses), args.advRatio)

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            if ADV_TRAINING:
                inputs = adv_transform_inputs(inputs, targets)
                targets = adv_transform_labels(targets)

                import matplotlib.pyplot as plt
                img_grid = torchvision.utils.make_grid(inputs, nrow=5)
                print(targets)
                plt.imshow(img_grid.permute(1, 2, 0))
                plt.show()

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)

                if ADV_TRAINING and get_mode(args.nClasses) == "2n":
                    outputs = torch.remainder(outputs, 10)

                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc


    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()
