import os
import pathlib
import random
import shutil
import time
import json
import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from utils.conv_type import STRConv

from args import args

import data
import models

use_cuda = torch.cuda.is_available()
in_activation = {}
out_activation = {}

def main():
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # create model
    model = get_model(args)

    # Loading pretrained model
    pretrained(args, model)

    data = get_dataset(args)

    for n, m in model.named_modules():
        if isinstance(m, STRConv):
            #print(n, m, m.getSparseWeight().shape)
            saveTensor(args, n, 'weight', m.getSparseWeight())
            m.register_forward_hook(get_activation(args, n, 'in'))
            m.register_forward_hook(get_activation(args, n, 'out'))
            #print(m.getSparseWeight())

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

            assert torch.equal(in_activation['conv1'], images)

    # check correctness
    with torch.no_grad():
        for n, m in model.named_modules():
            if isinstance(m, STRConv):
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

    save_dir = pathlib.Path(f"/data/sanchez/benchmarks/yifany/sconv/inputs/{args.arch+args.name}/{mode}")

    if not save_dir.exists():
        os.makedirs(save_dir)

    with (save_dir / "{}.mtx".format(name)).open('w') as fp:
        content = []

        content.append("%%MatrixMarket matrix coordinate real general")
        content.append("% {} tensor".format(mode))

        sizes = list(data.shape)
        content.append(" ".join([str(x) for x in sizes]))

        data = data.to_sparse()
        indices = data.indices().T.tolist()
        for idx in range(data.values().size()[0]): # only store coordinates
            coordinates = indices[idx]
            content.append(" ".join([str(x+1) for x in coordinates]))

        fp.write("\n".join(content))


def pretrained(args, model):
    assert args.pretrained

    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        if use_cuda:
            pretrained = torch.load(
                args.pretrained,
                map_location=torch.device("cuda:{}".format(args.multigpu[0])),
            )["state_dict"]
        else:
            pretrained = torch.load(
                args.pretrained,
                map_location=torch.device('cpu'),
            )["state_dict"]

        model_state_dict = model.state_dict()

        pretrained_final = {}
        for k, v in pretrained.items():
            if k.startswith("module."): # pretrained model store as dataparallel module
                key = k[7:]
            else:
                key = k
            if (key in model_state_dict and v.size() == model_state_dict[key].size()):
                pretrained_final[key] = v

        assert args.conv_type == "STRConv"

        model_state_dict.update(pretrained_final)
        model.load_state_dict(model_state_dict)

    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_model(args):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    print(f"=> Num model params {sum(p.numel() for p in model.parameters())}")

    return model


if __name__ == "__main__":
    main()