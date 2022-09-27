import os

def joinpath(a,b): return os.path.join(a, b)

def EnsureDirExists(dir):
    if not os.path.exists(dir):
        print("Creating %s" % dir)
        os.makedirs(dir)

# for ResNet

def processResNetLayers(layers):
    tmp = [filename[:-4] for filename in layers if filename.endswith('.tns')]
    a = []
    for layer in tmp:
        if "downsample" not in layer:
            a.append(layer)
        else:
            a.append(layer[:-2])
    tmp = a
    tmp.sort()
    return tmp

def getResNetSrcTensor(layer, benchmark_dir, mode):
    if "downsample" not in layer:
        srcTensor = joinpath(joinpath(benchmark_dir, mode), layer+".tns")
    else:
        srcTensor = joinpath(joinpath(benchmark_dir, mode), layer+".0.tns")
    return srcTensor

def getResNetSrcBN(layer, benchmark_dir):
    if "downsample" not in layer:
        newLayer = layer.replace('conv', 'bn')
        srcTensor = joinpath(joinpath(benchmark_dir, 'bn'), newLayer + ".txt")
    else:
        srcTensor = joinpath(joinpath(benchmark_dir, 'bn'), layer + ".1.txt")
    return srcTensor

# for VGG

def processVGGLayers(layers):
    tmp = [filename[:-4] for filename in layers if filename.endswith('.tns')]
    tmp.sort()
    return tmp

def getVGGSrcTensor(layer, benchmark_dir, mode):
    srcTensor = joinpath(joinpath(benchmark_dir, mode), layer+".tns")
    return srcTensor

def getVGGSrcBN(layer, benchmark_dir):
    layername = layer.split(".")[:-1]
    layerid = int(layer.split(".")[-1])
    layername.append(str(layerid+1))
    newLayer = ".".join(layername)
    srcTensor = joinpath(joinpath(benchmark_dir, 'bn'), newLayer + ".txt")
    return srcTensor

# for GoogLeNet

def getGoogLeNetSrcBN(layer, benchmark_dir):
    newLayer = layer.replace('conv', 'bn')
    srcTensor = joinpath(joinpath(benchmark_dir, 'bn'), newLayer + ".txt")
    return srcTensor 

def linktensor(network):

    if network.startswith("ResNet"):
        func = [processResNetLayers, getResNetSrcTensor, getResNetSrcBN]
    elif network.startswith("VGG"):
        func = [processVGGLayers, getVGGSrcTensor, getVGGSrcBN]
    elif network.startswith("GoogLeNet"):
        func = [processVGGLayers, getVGGSrcTensor, getGoogLeNetSrcBN]

    #benchmark_dir = "/data/sanchez/benchmarks/yifany/sconv/inputs/ResNet50STR_98.98"
    benchmark_dir = "/data/scratch/yifany/sconv/inputs/%s" % (network)

    layers = []

    for (_, _, filenames) in os.walk(os.path.join(benchmark_dir, "weight")):
        layers.extend(filenames)

    layers = func[0](layers)

    path = "/data/sanchez/users/yifany/merge_tensor_prj/taco_apps/input/%s" % (network)
    EnsureDirExists(path)

    for i, layer in enumerate(layers):
        EnsureDirExists(joinpath(path, layer))
        for mode in ['in', 'weight', 'out']:
            tensor = joinpath(joinpath(path, layer), mode + '.tns')
            srcTensor = func[1](layer, benchmark_dir, mode)
            os.system("ln -s %s %s" % (srcTensor, tensor))
        # now link bn
        tensor = joinpath(joinpath(path, layer), 'bn.txt')
        srcTensor = func[2](layer, benchmark_dir)
        os.system("ln -s %s %s" % (srcTensor, tensor))

linktensor("ResNet50STR_98.98")
linktensor("ResNet50STR_98.05")
linktensor("ResNet50STR_96.11")
linktensor("ResNet50STR_95.15")
linktensor("ResNet50STR_90.23")
linktensor("ResNet50STR_81.27")

#linktensor("VGG16_BNDefault")
#linktensor("VGG16_BN_90")

#linktensor("GoogLeNetDefault")
