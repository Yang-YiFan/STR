import os
import pathlib
import random
import shutil
import time
import json

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
            print(n, m, m.getSparseWeight().shape)
            saveTensor(args, n, m.getSparseWeight())
            #print(m.getSparseWeight())

def saveTensor(args, name, data):
    save_dir = pathlib.Path(f"inputs/weight/{args.arch+args.name}")

    if not save_dir.exists():
        os.makedirs(save_dir)

    with (save_dir / "%s.mtx" % name).open('w') as fp:
        fp.write("%%MatrixMarket matrix coordinate real general\n")
        fp.write("% tensor\n")
        fp.write(" ".join([str(x) for x in list(data.shape)]) + "\n")
        

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
