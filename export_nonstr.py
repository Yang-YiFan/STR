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

from args import args

import data
import torchvision.models as models

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
            #print(n, m, m.getSparseWeight().shape)
            saveTensor(args, n, 'weight', m.getSparseWeight())
            handle1 = m.register_forward_hook(get_activation(args, n, 'in'))
            handle2 = m.register_forward_hook(get_activation(args, n, 'out'))
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


def get_activation(args, name, mode):
    def in_hook(model, input, output):
        in_activation[name] = input[0].detach()
        saveTensor(args, name, mode, input[0].detach())
    def out_hook(model, input, output):
        out_activation[name] = output.detach()
        saveTensor(args, name, mode, output.detach())
    if mode == 'in':
        return in_hook
    elif mode == 'out':
        return out_hook
    else:
        assert False

def saveTensor(args, name, mode, data):
    assert mode in ['weight', 'in', 'out']
    print("saving", name, mode, data.shape)

    #save_dir = pathlib.Path(f"/data/sanchez/benchmarks/yifany/sconv/inputs/{args.arch+args.name}/{mode}")
    save_dir = pathlib.Path(f"{base_dir}/{mode}")

    if not save_dir.exists():
        os.makedirs(save_dir)

    with (save_dir / "{}.info".format(name)).open('w') as fp:
        content = []

        content.append("# Metadata for the tensor")
        content.append("# type {} tensor".format(mode))

        content.append("# nnz " + str(data.to_sparse().values().size()[0]) + " sparsity ratio " + str(1.0 - data.to_sparse().values().size()[0] / data.reshape(-1).size()[0]))

        sizes = list(data.shape)
        content.append("# uncompressed shape "+" ".join([str(x) for x in sizes]))

        fp.write("\n".join(content))

    with (save_dir / "{}.tns".format(name)).open('w') as fp:
        content = []

        data = data.to_sparse()
        indices = data.indices().T.tolist()
        for idx in range(data.values().size()[0]): # store coordinates first, then values
            coordinates = indices[idx]
            content.append(" ".join([str(x+1) for x in coordinates]) + " " + str(data.values()[idx].item())) # coordinates start at 1

        fp.write("\n".join(content))

def saveBn(args, name, data): # data = [weight, bias, running_mean, running_var, eps]

    save_dir = pathlib.Path(f"{base_dir}/bn")

    if not save_dir.exists():
        os.makedirs(save_dir)

    with (save_dir / "{}.txt".format(name)).open('w') as fp:
        size = len(data[0])
        content = []

        content.append(str(size))
        weight = []
        bias = []
        for i in range(size):
            weight.append(1.0 * data[0][i] / math.sqrt(data[3][i] + data[4]))
            bias.append(data[1][i] - 1.0 * data[0][i] * data[2][i] / math.sqrt(data[3][i] + data[4]))

        tensors = [weight, bias]
        for tensor in tensors:
            tensor1 = [str(x) for x in tensor]
            content += tensor1

        fp.write("\n".join(content))


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_model(args):
    if args.arch == "AlexNet":
        model = models.alexnet(pretrained=True)

    return model


if __name__ == "__main__":
    main()
