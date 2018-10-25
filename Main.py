import torch #Import Torch Deep Learning Framework
from torch import nn, optim #Import Neural Networks and Optimisation packages

from DCGAN import DCGANGenerator, DCGANDiscriminator

print("   Optimal Material Generator using Generative Adversarial Networks   ")
print("                    Developed by ***REMOVED*** (BSc)                    ")
print("In fulfilment of Doctor of Engineering at the University of Nottingham")
print("----------------------------------------------------------------------")
print()

print("Running hardware checks...")

deviceCount = torch.cuda.device_count()
print("\tGPU devices available: " + str(deviceCount))

if deviceCount > 1:
    print("\t\tParallel Processing enabled")
    parallelProcessing = True
    # For information on parallel processing with PyTorch: https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
else:
    parallelProcessing = False
    print("\t\tParallel processing is unavailable")

if torch.cuda.is_available():
    print("\t\tCUDA processing is available!")

    devices = [deviceCount]

    for x in range(0, deviceCount):
        device = "cuda:" + str(x)
        devices[x] = torch.device(device)
        print("\t\t\tEnabled device '" + device + "'")

else:
    print("\t\tCUDA processing is unavailable!")
    devices = [1]
    devices[0] = torch.device("cpu")
    print("\t\t\tEnabled device 'CPU'")

print("Initialising Generator Adversarial Network")
discriminator = DCGANDiscriminator()
generator = DCGANGenerator()