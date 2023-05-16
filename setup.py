import os

def joinpath(a,b): return os.path.join(a, b)

def EnsureDirExists(dir):
    if not os.path.exists(dir):
        print("Creating %s" % dir)
        os.makedirs(dir)

# for ResNet

def processResNetLayers(layers):
    tmp = [filename[:-4] for filename in layers if filename.endswith('.tns')]
    unaryLayers = []
    binaryLayers = []
    for layer in tmp:
        if "add" in layer:
            binaryLayers.append(layer)
        else:
            unaryLayers.append(layer)
    unaryLayers.sort()
    binaryLayers.sort()
    return unaryLayers, binaryLayers

def getResNetSrcTensor(layer, benchmark_dir, mode, suffix, isUnary):
    if isUnary:
        srcTensor = joinpath(joinpath(benchmark_dir, mode), f"{layer}.tns")
    else:
        if suffix == "":
            srcTensor = joinpath(joinpath(benchmark_dir, mode), f"{layer}.tns")
        else:
            srcTensor = joinpath(joinpath(benchmark_dir, mode), f"{layer}.{suffix}.tns")
    return srcTensor

def getResNetSrcBN(layer, benchmark_dir):
    if "downsample" in layer:
        newLayer = layer[:-1] + "1"
        srcTensor = joinpath(joinpath(benchmark_dir, 'bn'), f"{newLayer}.txt")
    elif "add" in layer:
        # don't have bn for add, put a random name
        srcTensor = joinpath(joinpath(benchmark_dir, 'bn'), f"{layer}.txt")
    else:
        newLayer = layer.replace('conv', 'bn')
        srcTensor = joinpath(joinpath(benchmark_dir, 'bn'), f"{newLayer}.txt")
    return srcTensor

# for VGG

def processVGGLayers(layers):
    tmp = [filename[:-4] for filename in layers if filename.endswith('.tns')]
    tmp.sort()
    return tmp, []

def getVGGSrcTensor(layer, benchmark_dir, mode, suffix, isUnary):
    srcTensor = joinpath(joinpath(benchmark_dir, mode), f"{layer}.tns")
    return srcTensor

def getVGGSrcBN(layer, benchmark_dir):
    layername = layer.split(".")[:-1]
    layerid = int(layer.split(".")[-1])
    layername.append(str(layerid+1))
    newLayer = ".".join(layername)
    srcTensor = joinpath(joinpath(benchmark_dir, 'bn'), f"{newLayer}.txt")
    return srcTensor

# for GoogLeNet

def getGoogLeNetSrcBN(layer, benchmark_dir):
    newLayer = layer.replace('conv', 'bn')
    srcTensor = joinpath(joinpath(benchmark_dir, 'bn'), f"{newLayer}.txt")
    return srcTensor

# for MobileNetV1

def getMobileNetV1SrcBN(layer, benchmark_dir):
    newLayer = layer
    if layer[-1].isnumeric(): newLayer = layer[:-1] + str(int(layer[-1])+1)
    srcTensor = joinpath(joinpath(benchmark_dir, 'bn'), f"{newLayer}.txt")
    return srcTensor

def linktensor(network):

    if network.startswith("ResNet"):
        func = [processResNetLayers, getResNetSrcTensor, getResNetSrcBN]
    elif network.startswith("VGG"):
        func = [processVGGLayers, getVGGSrcTensor, getVGGSrcBN]
    elif network.startswith("GoogLeNet"):
        func = [processVGGLayers, getVGGSrcTensor, getGoogLeNetSrcBN]
    elif network.startswith("MobileNetV1"):
        func = [processVGGLayers, getVGGSrcTensor, getMobileNetV1SrcBN]
    else:
        assert False, "unsupported network!"

    #benchmark_dir = "/data/sanchez/benchmarks/yifany/sconv/inputs/ResNet50STR_98.98"
    benchmark_dir = f"/data/scratch/yifany/sconv/inputs/{network}"

    layers = []

    for (_, _, filenames) in os.walk(os.path.join(benchmark_dir, "out")):
        layers.extend(filenames)

    unaryLayers, binaryLayers = func[0](layers)

    path = f"/data/sanchez/users/yifany/merge_tensor_prj/taco_apps/input/{network}"
    EnsureDirExists(path)

    # handle unary first
    for i, layer in enumerate(unaryLayers):
        EnsureDirExists(joinpath(path, layer.replace(".", "_")))
        for mode, suffix in [('in', '0'), ('weight', ''), ('out', '')]:
            tensor = joinpath(joinpath(path, layer.replace(".", "_")), f'{mode}{suffix}.tns')
            srcTensor = func[1](layer, benchmark_dir, mode, suffix, True)
            os.system(f"ln -s {srcTensor} {tensor}")
        # now link bn
        tensor = joinpath(joinpath(path, layer.replace(".", "_")), 'bn.txt')
        srcTensor = func[2](layer, benchmark_dir)
        os.system(f"ln -s {srcTensor} {tensor}")

    # handle binary next
    for i, layer in enumerate(binaryLayers):
        EnsureDirExists(joinpath(path, layer.replace(".", "_")))
        for mode, suffix in [('in', '0'), ('in', '1'), ('out', '')]:
            tensor = joinpath(joinpath(path, layer.replace(".", "_")), f'{mode}{suffix}.tns')
            srcTensor = func[1](layer, benchmark_dir, mode, suffix, False)
            os.system(f"ln -s {srcTensor} {tensor}")
        # now link bn
        tensor = joinpath(joinpath(path, layer.replace(".", "_")), 'bn.txt')
        srcTensor = func[2](layer, benchmark_dir)
        os.system(f"ln -s {srcTensor} {tensor}")

linktensor("ResNet50STR_98.98")
#linktensor("ResNet50STR_98.05")
#linktensor("ResNet50STR_96.11")
#linktensor("ResNet50STR_95.15")
#linktensor("ResNet50STR_90.23")
#linktensor("ResNet50STR_81.27")

#linktensor("VGG16_BNDefault")
#linktensor("VGG16_BN_90")

#linktensor("GoogLeNetDefault")

#linktensor("MobileNetV1STR_89.01")
#linktensor("MobileNetV1STR_75.28")
