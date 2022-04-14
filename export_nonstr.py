import os
import pathlib
import random
import shutil
import time
import json
import tqdm
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from utils.conv_type import sparseFunction

from args import args

import data
import torchvision.models as models

from export import get_activation, saveTensor, saveBn, get_dataset

use_cuda = torch.cuda.is_available()
in_activation = {}
out_activation = {}
torch.set_num_threads(16)
base_dir = f"/scratch/yifany/sconv/inputs/{args.arch+args.name}"

def main():
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # create model
    model = get_model(args)

    data = get_dataset(args)

    # export BN
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            print(n, m)
            saveBn(args, n, [m.weight.tolist(), m.bias.tolist(), m.running_mean.tolist(), m.running_var.tolist(), m.eps])

    hooks = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            #print(n, m, m.weight.shape)
            saveTensor(args, n, 'weight', sparseFunction(m.weight, torch.tensor([-100]))) # alexnet have bias, ignore it for now
            handle1 = m.register_forward_hook(get_activation(args, n, 'in', in_activation, out_activation))
            handle2 = m.register_forward_hook(get_activation(args, n, 'out', in_activation, out_activation))
            #print(m.getSparseWeight())
            hooks.append(handle1)
            hooks.append(handle2)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in tqdm.tqdm(
            enumerate(data.val_loader), ascii=True, total=len(data.val_loader)
        ):
            if i > 0: break

            if use_cuda:
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)

                target = target.cuda(args.gpu, non_blocking=True).long()

            # compute output
            output = model(images)

            #assert torch.equal(in_activation['conv1'], images)

    # remove all hooks
    for hook in hooks:
        hook.remove()

    # check correctness
    with torch.no_grad():
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                assert torch.equal(m(in_activation[n]), out_activation[n])


def get_model(args):
    if args.arch == "AlexNet":
        model = models.alexnet(pretrained=True)
        assert False # somehow the export check fails for alexnet
    elif args.arch == "VGG16_BN":
        model = models.vgg16_bn(pretrained=True)

    return model


if __name__ == "__main__":
    main()
