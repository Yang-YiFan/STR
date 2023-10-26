import os
import pathlib
import random
import shutil
import time
import json
import tqdm
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from utils.conv_type import STRConv
from models.resnet import MyAdd

from setup import EnsureDirExists

from args import args

import data
import models

use_cuda = torch.cuda.is_available()
# for native conv
in_activation = {}
out_activation = {}
# for im2col
in_activation_matrix = {}
out_activation_matrix = {}
weight_matrix = {}
torch.set_num_threads(16)
base_dir = f"/scratch/yifany/sconv/inputs/{args.arch+args.name}"
matrix_dir = f"/scratch/yifany/spmspm/inputs/{args.arch+args.name}"

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
            first_layer = n
            break

    # export BN
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            print(n, m)
            saveBn(args, n, [m.weight.tolist(), m.bias.tolist(), m.running_mean.tolist(), m.running_var.tolist(), m.eps])

    for mode in ['weight', 'in', 'out']:
        EnsureDirExists(os.path.join(matrix_dir, mode))

    hooks = []
    # export conv and FC
    for n, m in model.named_modules():
        if isinstance(m, STRConv) or isinstance(m, MyAdd):
            #print(n, m, m.getSparseWeight().shape)
            if isinstance(m, STRConv):
                saveTensor(args, n, 'weight', m.getSparseWeight())
                sparse_weight = m.getSparseWeight().detach() # (K, C, R, S)
                K = sparse_weight.shape[0]
                sparse_weight = sparse_weight.view(K, -1) # (K, C*R*S)
                sparse_weight = sparse_weight.permute(1, 0).contiguous() # (C*R*S, K)
                print(n, m, sparse_weight.shape)
                weight_matrix[n] = sparse_weight
                np.save(f"{matrix_dir}/weight/{n}.npy", sparse_weight.cpu().numpy())

            handle1 = m.register_forward_hook(get_activation(args, n, 'in', in_activation, out_activation))
            handle2 = m.register_forward_hook(get_activation(args, n, 'out', in_activation, out_activation))
            handle3 = m.register_forward_hook(saveMatrix(args, n, 'in', in_activation_matrix, out_activation_matrix, STRConv))
            handle4 = m.register_forward_hook(saveMatrix(args, n, 'out', in_activation_matrix, out_activation_matrix, STRConv))
            #print(m.getSparseWeight())
            hooks.append(handle1)
            hooks.append(handle2)
            hooks.append(handle3)
            hooks.append(handle4)

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

            if first_layer in in_activation.keys():
                assert torch.equal(in_activation[first_layer], images)

    # remove all hooks
    for hook in hooks:
        hook.remove()

    # check correctness
    with torch.no_grad():
        for n, m in model.named_modules():
            if isinstance(m, STRConv) and n in in_activation.keys() and n in out_activation.keys():
                assert torch.equal(m(in_activation[n]), out_activation[n])
            if isinstance(m, STRConv) and n in in_activation_matrix.keys() and n in out_activation_matrix.keys() and n in weight_matrix.keys():
                out = torch.matmul(in_activation_matrix[n], weight_matrix[n])
                assert torch.allclose(out, out_activation_matrix[n], rtol=1e-02, atol=1e-03)


def get_activation(args, name, mode, in_activation, out_activation, unsqueeze=False):
    def in_hook(model, input, output):
        if len(input) == 1: # if only 1 input, handle separately for legacy reasons
            in_activation[name] = input[0].detach()
            saveTensor(args, name, mode, input[0].detach(), unsqueeze)
        else:
            for i in range(len(input)):
                in_activation[name+"."+str(i)] = input[i].detach()
                saveTensor(args, name+"."+str(i), mode, input[i].detach(), unsqueeze)
    def out_hook(model, input, output):
        out_activation[name] = output.detach()
        saveTensor(args, name, mode, output.detach(), unsqueeze)
    if mode == 'in':
        return in_hook
    elif mode == 'out':
        return out_hook
    else:
        assert False

def saveMatrix(args, name, mode, in_activation, out_activation, convType):
    def in_hook(model, input, output):
        for i in range(len(input)):
            tensor = input[i].detach() # (N, C, H, W)
            if isinstance(model, convType):
                assert(len(input) == 1)
                tensor = nn.functional.unfold(tensor, model.kernel_size, padding=model.padding, stride=model.stride) # (N, C*R*S, H*W)
                tensor = tensor.permute(0, 2, 1).contiguous() # (N, H*W, C*R*S)
                in_activation[name] = tensor
                B = min(256, max(1, int(2 ** (round(math.log2(8000 / tensor.shape[1])))))) # do multiple batch so that number of row is roughly 8000
                assert B <= 256
                tensor = tensor.view(-1, B * tensor.shape[1], tensor.shape[2]) # save a batch of B (N/B, B*H*W, C*R*S)
                print(name, mode, model, B, tensor.shape)
            np.save(f"{matrix_dir}/{mode}/{name}.npy", tensor[0].cpu().numpy()) # no batch dim
    def out_hook(model, input, output):
        tensor = output.detach() # (N, K, H, W)
        if isinstance(model, convType):
            tensor = tensor.view(tensor.shape[0], tensor.shape[1], -1) # (N, K, H*W)
            tensor = tensor.permute(0, 2, 1).contiguous() # (N, H*W, K)
            out_activation[name] = tensor
            B = min(256, max(1, int(2 ** (round(math.log2(8000 / tensor.shape[1])))))) # do multiple batch so that number of row is roughly 8000
            assert B <= 256
            tensor = tensor.view(-1, B * tensor.shape[1], tensor.shape[2]) # save a batch of B (N/B, B*H*W, K)
            print(name, mode, model, B, tensor.shape)
        np.save(f"{matrix_dir}/{mode}/{name}.npy", tensor[0].cpu().numpy()) # no batch dim
    if mode == 'in':
        return in_hook
    elif mode == 'out':
        return out_hook
    else:
        assert False

def saveTensor(args, name, mode, data, unsqueeze=False):
    assert mode in ['weight', 'in', 'out']
    # for FC make the tensor shape like 1x1 conv
    if unsqueeze:
        data = torch.unsqueeze(torch.unsqueeze(data, -1), -1)
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
        content.append(f"# ndim {len(sizes)} uncompressed shape "+" ".join([str(x) for x in sizes]))

        fp.write("\n".join(content))

    open(save_dir / "{}.tns".format(name), 'w').close()

    with (save_dir / "{}.tns".format(name)).open('a') as fp:
        content = []

        data = data.to_sparse()
        crd = data.indices().T
        val = data.values()
        batch = 10000
        size = data.values().size()[0]
        for idx in range(size): # store coordinates first, then values
            if idx % batch == 0:
                fp.write("\n".join(content))
                fp.flush()
                if idx == 0:
                    content = []
                else:
                    content = [""]
                begin = idx
                end = min(idx + batch, size)
                local_crd = crd[begin:end].tolist() # sw caching
                local_val = val[begin:end].tolist() # sw caching

            content.append(" ".join([str(x+1) for x in local_crd[idx % batch]]) + " " + str(local_val[idx % batch])) # coordinates start at 1

        if len(content) > 0:
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
